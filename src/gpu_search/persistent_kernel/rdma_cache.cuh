#pragma once

#include "gpu_search/persistent_kernel/candidate_scoring.cuh"

namespace gpu_search::persistent_kernel_detail {

__device__ __forceinline__ f32 storage_component(
    const PersistentKernelParams& params, const u8* vector, u32 dimension) {
  if (params.vector_dtype == 0) return reinterpret_cast<const f32*>(vector)[dimension];
  if (params.vector_dtype == 1) return static_cast<f32>(vector[dimension]);
  return static_cast<f32>(reinterpret_cast<const int8_t*>(vector)[dimension]);
}

__device__ f32 exact_storage_distance(const PersistentKernelParams& params,
                                      const f32* query, const u8* vector) {
  f32 distance = 0.0f;
  for (u32 dimension = 0; dimension < params.dim; ++dimension) {
    const f32 component = storage_component(params, vector, dimension);
    const f32 difference = query[dimension] - component;
    distance += difference * difference;
  }
  return distance;
}

__device__ f32 exact_anchor_distance(const PersistentKernelParams& params,
                                     const f32* query, u32 anchor) {
  f32 distance = 0.0f;
  for (u32 dimension = 0; dimension < params.dim; ++dimension) {
    const f32 component = params.anchor_vectors[
      static_cast<size_t>(dimension) * params.anchor_count + anchor];
    const f32 difference = query[dimension] - component;
    distance += difference * difference;
  }
  return distance;
}

__device__ i32 direct_fetch(const PersistentKernelParams& params,
                            u32 memory_node, u64 remote_offset,
                            u8* destination, u32 bytes, u32 lane) {
#ifdef DVSTOR_HAVE_GPUNETIO
  if (memory_node >= params.direct_region_count || params.direct_qps == nullptr ||
      params.direct_qp_locks == nullptr || params.direct_qps_per_node == 0 ||
      params.direct_disabled == nullptr ||
      *reinterpret_cast<volatile u32*>(params.direct_disabled) != 0) return -EHOSTDOWN;
  const u32 qp_index = (lane % params.direct_qps_per_node) *
    params.direct_region_count + memory_node;
  if (params.direct_qps[qp_index] == nullptr) return -EINVAL;
  auto* qp = reinterpret_cast<doca_gpu_dev_verbs_qp*>(params.direct_qps[qp_index]);
  const DirectRemoteRegion& region = params.direct_regions[memory_node];
  doca_gpu_dev_verbs_addr remote{.addr = region.address + remote_offset, .key = region.rkey};
  doca_gpu_dev_verbs_addr local{
    .addr = reinterpret_cast<u64>(destination) - params.direct_local_iova_base,
    .key = params.direct_local_mkey,
  };
  doca_gpu_dev_verbs_addr dump{
    .addr = reinterpret_cast<u64>(params.direct_dump) - params.direct_local_iova_base,
    .key = params.direct_local_mkey,
  };
  i32 status = lock_direct_qp(params.direct_qp_locks + qp_index, params.stop,
                              params.direct_disabled);
  if (status != 0) {
    if (params.direct_error != nullptr) atomicCAS(params.direct_error, 0, status);
    if (status != -ECANCELED) atomicExch(params.direct_disabled, 1u);
    return status;
  }
  if (bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
    if (params.direct_error != nullptr) atomicCAS(params.direct_error, 0, -E2BIG);
    atomicExch(params.direct_disabled, 1u);
    unlock_direct_qp(params.direct_qp_locks + qp_index);
    return -E2BIG;
  }
  const doca_gpu_dev_verbs_ticket_t read_ticket = qp->sq_wqe_pi;
  auto* completion_queue = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
  const doca_gpu_dev_verbs_ticket_t completion_ticket =
    doca_gpu_dev_verbs_load_relaxed<
      DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
        &completion_queue->cqe_ci);
  auto* read_wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, read_ticket);
  const bool need_dump = qp->need_dump;
  doca_gpu_dev_verbs_wqe_prepare_read(
    qp, read_wqe, read_ticket,
    need_dump ? DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE
              : DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
    remote.addr, remote.key, local.addr, local.key, bytes);
  doca_gpu_dev_verbs_ticket_t final_ticket = read_ticket;
  if (need_dump) {
    final_ticket = read_ticket + 1;
    auto* dump_wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, final_ticket);
    doca_gpu_dev_verbs_wqe_prepare_dump(
      qp, dump_wqe, final_ticket, DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
      dump.addr, dump.key, 1);
  }
  doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
    qp, final_ticket + 1);
  status = poll_direct_cq(completion_queue, completion_ticket,
                          params.direct_timeout_ns, params.stop,
                          params.direct_disabled);
  if (status != 0) {
    if (params.direct_error != nullptr) atomicCAS(params.direct_error, 0, status);
    atomicExch(params.direct_disabled, 1u);
  }
  unlock_direct_qp(params.direct_qp_locks + qp_index);
  return status;
#else
  (void)params;
  (void)memory_node;
  (void)remote_offset;
  (void)destination;
  (void)bytes;
  (void)lane;
  return -ENOTSUP;
#endif
}

__device__ i32 direct_fetch_batch(const PersistentKernelParams& params,
                                  u32 memory_node, const u32* request_shards,
                                  const u64* remote_offsets, u32 request_count,
                                  u8* destination_base, u32 destination_stride,
                                  u32 bytes, u32 lane,
                                  const u64* local_iova_offsets = nullptr,
                                  i32* owner_completion = nullptr,
                                  bool defer_owner_wait = false,
                                  u32* owner_progress = nullptr) {
#ifdef DVSTOR_HAVE_GPUNETIO
  u32 matching = 0;
  for (u32 index = 0; index < request_count; ++index) {
    if (request_shards[index] == memory_node) ++matching;
  }
  if (matching == 0) return 0;
  if (memory_node >= params.direct_region_count || params.direct_qps == nullptr ||
      params.direct_qp_locks == nullptr || params.direct_qps_per_node == 0 ||
      params.direct_disabled == nullptr ||
      *reinterpret_cast<volatile u32*>(params.direct_disabled) != 0) return -EHOSTDOWN;
  const u32 qp_index = (lane % params.direct_qps_per_node) *
    params.direct_region_count + memory_node;
  if (params.direct_qps[qp_index] == nullptr) return -EINVAL;
  if (params.direct_batch_queues != nullptr && owner_completion != nullptr) {
    if (qp_index >= params.direct_batch_queue_count) return -EINVAL;
    const u64 started = global_time_ns();
    if (owner_progress != nullptr) {
      *reinterpret_cast<volatile u32*>(owner_progress) = 2;
      __threadfence_system();
    }
    atomicExch(owner_completion, -EINPROGRESS);
    __threadfence();
    const DirectBatchDescriptor descriptor{
      .request_shards = request_shards,
      .remote_offsets = remote_offsets,
      .local_iova_offsets = local_iova_offsets,
      .completion_status = owner_completion,
      .request_count = request_count,
      .memory_node = memory_node,
      .bytes = bytes,
    };
    while (!device_ring_try_push(params.direct_batch_queues[qp_index], descriptor)) {
      if (*reinterpret_cast<const volatile u32*>(params.stop) != 0) {
        atomicExch(owner_completion, -ECANCELED);
        return -ECANCELED;
      }
      if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) != 0) {
        atomicExch(owner_completion, -EHOSTDOWN);
        return -EHOSTDOWN;
      }
      if (global_time_ns() - started >= params.direct_timeout_ns) {
        atomicExch(owner_completion, -ETIMEDOUT);
        if (params.direct_error != nullptr) {
          atomicCAS(params.direct_error, 0, -ETIMEDOUT);
        }
        atomicExch(params.direct_disabled, 1u);
        return -ETIMEDOUT;
      }
      device_ring_relax(128);
    }
    if (owner_progress != nullptr) {
      *reinterpret_cast<volatile u32*>(owner_progress) = 3;
      __threadfence_system();
    }
    if (defer_owner_wait) return -EINPROGRESS;
    for (;;) {
      const i32 status = *reinterpret_cast<volatile i32*>(owner_completion);
      if (status != -EINPROGRESS) return status;
      if (*reinterpret_cast<const volatile u32*>(params.stop) != 0) return -ECANCELED;
      if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) != 0) {
        return -EHOSTDOWN;
      }
      if (global_time_ns() - started >= params.direct_timeout_ns) {
        if (params.direct_error != nullptr) {
          atomicCAS(params.direct_error, 0, -ETIMEDOUT);
        }
        atomicExch(params.direct_disabled, 1u);
        return -ETIMEDOUT;
      }
      device_ring_relax(128);
    }
  }
  auto* qp = reinterpret_cast<doca_gpu_dev_verbs_qp*>(params.direct_qps[qp_index]);
  if (bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE ||
      matching + (qp->need_dump ? 1u : 0u) > qp->sq_wqe_num) return -E2BIG;
  i32 status = lock_direct_qp(params.direct_qp_locks + qp_index, params.stop,
                              params.direct_disabled);
  if (status != 0) return status;
  const DirectRemoteRegion& region = params.direct_regions[memory_node];
  auto* completion_queue = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
  const doca_gpu_dev_verbs_ticket_t completion_ticket =
    doca_gpu_dev_verbs_load_relaxed<
      DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
        &completion_queue->cqe_ci);
  const doca_gpu_dev_verbs_ticket_t first_wqe = qp->sq_wqe_pi;
  u32 posted = 0;
  for (u32 index = 0; index < request_count; ++index) {
    if (request_shards[index] != memory_node) continue;
    const doca_gpu_dev_verbs_ticket_t ticket = first_wqe + posted;
    auto* wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, ticket);
    const bool last_read = posted + 1 == matching;
    const auto flags = !qp->need_dump && last_read
      ? DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE
      : DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;
    const u64 local_iova = local_iova_offsets != nullptr
      ? local_iova_offsets[index]
      : reinterpret_cast<u64>(destination_base) +
          static_cast<u64>(index) * destination_stride - params.direct_local_iova_base;
    doca_gpu_dev_verbs_wqe_prepare_read(
      qp, wqe, ticket, flags, region.address + remote_offsets[index], region.rkey,
      local_iova, params.direct_local_mkey, bytes);
    ++posted;
  }
  doca_gpu_dev_verbs_ticket_t final_wqe = first_wqe + posted - 1;
  if (qp->need_dump) {
    final_wqe = first_wqe + posted;
    auto* dump_wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, final_wqe);
    doca_gpu_dev_verbs_wqe_prepare_dump(
      qp, dump_wqe, final_wqe, DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
      reinterpret_cast<u64>(params.direct_dump) - params.direct_local_iova_base,
      params.direct_local_mkey, 1);
  }
  doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
    qp, final_wqe + 1);
  status = poll_direct_cq(completion_queue, completion_ticket,
                          params.direct_timeout_ns, params.stop,
                          params.direct_disabled);
  if (status != 0) {
    if (params.direct_error != nullptr) atomicCAS(params.direct_error, 0, status);
    atomicExch(params.direct_disabled, 1u);
  }
  unlock_direct_qp(params.direct_qp_locks + qp_index);
  return status;
#else
  (void)params;
  (void)memory_node;
  (void)request_shards;
  (void)remote_offsets;
  (void)request_count;
  (void)destination_base;
  (void)destination_stride;
  (void)bytes;
  (void)lane;
  (void)local_iova_offsets;
  (void)owner_completion;
  (void)owner_progress;
  return -ENOTSUP;
#endif
}

__device__ i32 wait_direct_batch(const PersistentKernelParams& params,
                                 i32* owner_completion) {
#ifdef DVSTOR_HAVE_GPUNETIO
  if (owner_completion == nullptr) return -EINVAL;
  const u64 started = global_time_ns();
  for (;;) {
    const i32 status = *reinterpret_cast<volatile i32*>(owner_completion);
    if (status != -EINPROGRESS) return status;
    if (*reinterpret_cast<const volatile u32*>(params.stop) != 0) {
      return -ECANCELED;
    }
    if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) != 0) {
      return -EHOSTDOWN;
    }
    if (global_time_ns() - started >= params.direct_timeout_ns) {
      if (params.direct_error != nullptr) {
        atomicCAS(params.direct_error, 0, -ETIMEDOUT);
      }
      atomicExch(params.direct_disabled, 1u);
      return -ETIMEDOUT;
    }
    device_ring_relax(128);
  }
#else
  (void)params;
  (void)owner_completion;
  return -ENOTSUP;
#endif
}

__device__ bool exact_record_visible(const u8* record) {
  const u64 header = *reinterpret_cast<const u64*>(record);
  return (header & (kNodeLockMask | kNodeDeletedMask)) == 0;
}

__device__ bool admit_graph_cache(const PersistentKernelParams& params, u64 key) {
  if (params.graph_admission_sets == 0 || params.graph_admission_keys == nullptr ||
      params.graph_admission_victims == nullptr) return true;
  const u32 set = hash64(key) % params.graph_admission_sets;
  const u32 base = set * kCacheAdmissionWays;
  for (u32 way = 0; way < kCacheAdmissionWays; ++way) {
    if (load_cg(params.graph_admission_keys + base + way) == key) return true;
  }
  const u32 way = atomicAdd(params.graph_admission_victims + set, 1u) %
    kCacheAdmissionWays;
  atomicExch(reinterpret_cast<unsigned long long*>(
               params.graph_admission_keys + base + way), key);
  return false;
}

__device__ bool admit_exact_cache(const PersistentKernelParams& params, u32 key) {
  if (params.exact_admission_sets == 0 || params.exact_admission_keys == nullptr ||
      params.exact_admission_victims == nullptr) return true;
  const u32 set = hash32(key) % params.exact_admission_sets;
  const u32 base = set * kCacheAdmissionWays;
  for (u32 way = 0; way < kCacheAdmissionWays; ++way) {
    if (load_cg(params.exact_admission_keys + base + way) == key) return true;
  }
  const u32 way = atomicAdd(params.exact_admission_victims + set, 1u) %
    kCacheAdmissionWays;
  atomicExch(params.exact_admission_keys + base + way, key);
  return false;
}

__device__ bool approximate_handles_batch(const PersistentKernelParams& params,
                                          const QueryDescriptor& descriptor,
                                          const f32* query_lut,
                                          u32* handles,
                                          u32 count,
                                          f32* distances) {
  __shared__ i32 shard_status[kPersistentMaxShards];
  __shared__ u32 failed;
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) * kPersistentMaxMergeCandidates;
  u32* request_shards = params.dynamic_code_request_shards + request_base;
  u64* request_offsets = params.dynamic_code_request_offsets + request_base;
  u64* request_local_iova_offsets =
    params.dynamic_code_request_local_iovas + request_base;
  if (threadIdx.x == 0) failed = 0;
  for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
    const u32 handle = handles[index];
    request_shards[index] = UINT32_MAX;
    request_offsets[index] = 0;
    request_local_iova_offsets[index] = 0;
    distances[index] = FLT_MAX;
    if (handle == UINT32_MAX) continue;
    if ((handle & kDeltaHandleBit) == 0) {
      distances[index] = approximate_handle(
        params, query_lut, handle, descriptor.snapshot_epoch);
      continue;
    }

    u64 raw = 0;
    u64 graph_offset = 0;
    u32 shard = 0;
    if (!resolve_handle(params, handle, raw, shard, graph_offset)) continue;
    const u32 delta_slot = delta_slot_from_raw(params, raw);
    const u32 delta_count = min(load_cg(params.delta_count), params.delta_capacity);
    if (delta_slot < delta_count &&
        params.delta_records[delta_slot].remote_node == raw) {
      const DeviceDeltaRecord& record = params.delta_records[delta_slot];
      if (delta_code_visible(record, descriptor.snapshot_epoch)) {
        distances[index] = approximate_entry(
          params, query_lut,
          params.delta_pq_codes +
            static_cast<size_t>(delta_slot) * params.pq_code_bytes);
        continue;
      }
      const u64 superseded = load_cg(&record.superseded_epoch);
      if (record.epoch <= descriptor.snapshot_epoch &&
          ((record.flags & kDeltaDeleted) != 0 ||
           (superseded != 0 && superseded <= descriptor.snapshot_epoch))) {
        continue;
      }
    }

    const u32 resident_slot = resident_pq_slot_from_raw(params, raw);
    if (resident_slot != UINT32_MAX && params.resident_pq_codes != nullptr) {
      distances[index] = approximate_entry(
        params, query_lut,
        params.resident_pq_codes +
          static_cast<size_t>(resident_slot) * params.pq_code_bytes);
      continue;
    }

    if (params.dynamic_code_records == nullptr || shard >= params.num_shards) continue;
    const u64 node_offset = (raw << 16) >> 16;
    request_shards[index] = shard;
    request_offsets[index] = node_offset + params.shards[shard].dynamic_code_offset;
    u8* destination = params.dynamic_code_records +
      (request_base + index) * params.pq_code_bytes;
    request_local_iova_offsets[index] =
      reinterpret_cast<u64>(destination) - params.direct_local_iova_base;
  }
  for (u32 shard = threadIdx.x; shard < params.num_shards; shard += blockDim.x) {
    shard_status[shard] = 0;
  }
  __syncthreads();

  for (u32 shard = threadIdx.x; shard < params.num_shards; shard += blockDim.x) {
    i32* owner_completion = params.direct_batch_statuses == nullptr ? nullptr :
      params.direct_batch_statuses +
        static_cast<size_t>(descriptor.query_slot) * params.num_shards + shard;
    shard_status[shard] = direct_fetch_batch(
        params, shard, request_shards, request_offsets, count,
        params.dynamic_code_records + request_base * params.pq_code_bytes,
        params.pq_code_bytes, params.pq_code_bytes,
        (descriptor.query_slot + shard) % params.direct_qps_per_node,
        request_local_iova_offsets, owner_completion, true);
  }
  __syncthreads();
  for (u32 shard = threadIdx.x; shard < params.num_shards; shard += blockDim.x) {
    if (shard_status[shard] != -EINPROGRESS) continue;
    i32* owner_completion = params.direct_batch_statuses +
      static_cast<size_t>(descriptor.query_slot) * params.num_shards + shard;
    shard_status[shard] = wait_direct_batch(params, owner_completion);
  }
  __syncthreads();

  for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
    const u32 shard = request_shards[index];
    if (shard == UINT32_MAX) continue;
    if (shard_status[shard] != 0) {
      atomicExch(&failed, 1u);
      continue;
    }
    const u8* code = params.dynamic_code_records +
      (request_base + index) * params.pq_code_bytes;
    distances[index] = approximate_entry(params, query_lut, code);
  }
  __syncthreads();
  return failed == 0;
}

__device__ void exactify_into_beam(const PersistentKernelParams& params,
                                   const QueryDescriptor& descriptor,
                                   const f32* query, u32* candidate_handles,
                                   u32* candidate_ids, f32* candidate_distances,
                                   u32 candidate_count, u32* beam_handles,
                                   u32* beam_ids, f32* beam_distances,
                                   u8* beam_expanded, u32& beam_count,
                                   u32* exact_reads, u32* exact_cache_hits,
                                   u32 beam_capacity, bool reset_beam,
                                   u32* merge_handles, u32* merge_ids,
                                   f32* merge_distances, u8* merge_expanded) {
    __shared__ u32 request_cache_slots[kPersistentMaxExact];
    __shared__ u32 request_delta_slots[kPersistentMaxExact];
    __shared__ u8 request_cache_owned[kPersistentMaxExact];
    __shared__ i32 shard_status[kPersistentMaxShards];
    const size_t request_base =
      static_cast<size_t>(descriptor.query_slot) * kPersistentMaxMergeCandidates;
    u32* request_shards = params.dynamic_code_request_shards + request_base;
    u64* request_offsets = params.dynamic_code_request_offsets + request_base;
    u64* request_local_iova_offsets =
      params.dynamic_code_request_local_iovas + request_base;
    for (u32 index = threadIdx.x; index < candidate_count; index += blockDim.x) {
      request_shards[index] = UINT32_MAX;
      request_offsets[index] = 0;
      request_cache_slots[index] = UINT32_MAX;
      request_delta_slots[index] = UINT32_MAX;
      request_cache_owned[index] = 0;
      candidate_ids[index] = UINT32_MAX;
      candidate_distances[index] = FLT_MAX;
      const u32 handle = candidate_handles[index];
      const bool dynamic = (handle & kDeltaHandleBit) != 0;
      if (!dynamic && base_overridden(params, handle, descriptor.snapshot_epoch)) continue;
      u64 raw = 0;
      u64 graph_offset = 0;
      u32 shard = 0;
      if (!resolve_handle(params, handle, raw, shard, graph_offset)) continue;
      if (dynamic) {
        const u32 delta_slot = delta_slot_from_raw(params, raw);
        const u32 delta_count = min(load_cg(params.delta_count), params.delta_capacity);
        if (delta_slot < delta_count &&
            params.delta_records[delta_slot].remote_node == raw &&
            delta_code_visible(params.delta_records[delta_slot],
                               descriptor.snapshot_epoch)) {
          request_delta_slots[index] = delta_slot;
          continue;
        }
      }
      request_offsets[index] = ((raw << 16) >> 16) + params.node_meta_offset;
      u32 cache_slot = UINT32_MAX;
      if (!dynamic && params.exact_cache_sets != 0 && params.exact_cache_ways != 0) {
        const u32 set = hash32(handle) % params.exact_cache_sets;
        bool cache_hit = false;
        for (u32 way = 0; way < params.exact_cache_ways; ++way) {
          const u32 slot = set * params.exact_cache_ways + way;
          const u32 state = *reinterpret_cast<volatile u32*>(
            params.exact_cache_states + slot);
          if (state == 2 && load_cg(params.exact_cache_keys + slot) == handle) {
            atomicAdd(params.exact_cache_readers + slot, 1u);
            __threadfence();
            if (*reinterpret_cast<volatile u32*>(params.exact_cache_states + slot) == 2 &&
                load_cg(params.exact_cache_keys + slot) == handle) {
              const u8* record = params.exact_cache +
                static_cast<size_t>(slot) * params.exact_cache_stride;
              if (exact_record_visible(record)) {
                candidate_ids[index] =
                  *reinterpret_cast<const u32*>(record + kNodeIdOffset);
                candidate_distances[index] = exact_storage_distance(
                  params, query, record + kNodeVectorOffset);
                atomicAdd(exact_cache_hits, 1u);
                cache_hit = true;
              }
            }
            atomicSub(params.exact_cache_readers + slot, 1u);
            if (cache_hit) break;
          }
        }
        if (cache_hit) continue;
        if (cache_slot == UINT32_MAX && admit_exact_cache(params, handle)) {
          const u32 start_way = atomicAdd(params.exact_cache_victims + set, 1u) %
            params.exact_cache_ways;
          for (u32 attempt = 0; attempt < params.exact_cache_ways; ++attempt) {
            const u32 slot = set * params.exact_cache_ways +
              (start_way + attempt) % params.exact_cache_ways;
            const u32 state = *reinterpret_cast<volatile u32*>(
              params.exact_cache_states + slot);
            if (state == 1 ||
                atomicCAS(params.exact_cache_states + slot, state, 1u) != state) {
              continue;
            }
            u32 wait = 0;
            while (*reinterpret_cast<volatile u32*>(params.exact_cache_readers + slot) != 0 &&
                   *reinterpret_cast<volatile u32*>(params.stop) == 0 &&
                   wait++ < kCacheWaitRounds) {
              device_ring_relax(128);
            }
            if (*reinterpret_cast<volatile u32*>(params.exact_cache_readers + slot) != 0) {
              atomicCAS(params.exact_cache_states + slot, 1u, state);
              continue;
            }
            params.exact_cache_keys[slot] = handle;
            __threadfence();
            cache_slot = slot;
            request_cache_owned[index] = 1;
            break;
          }
        }
      }
      request_cache_slots[index] = cache_slot;
      request_shards[index] = shard;
      const u8* destination = cache_slot != UINT32_MAX
        ? params.exact_cache + static_cast<size_t>(cache_slot) * params.exact_cache_stride
        : params.exact_records +
            (static_cast<size_t>(descriptor.query_slot) * params.exact_width + index) *
              params.node_record_bytes;
      request_local_iova_offsets[index] =
        reinterpret_cast<u64>(destination) - params.direct_local_iova_base;
    }
    for (u32 shard = threadIdx.x; shard < params.num_shards; shard += blockDim.x) {
      shard_status[shard] = 0;
    }
    __syncthreads();
    for (u32 shard = threadIdx.x; shard < params.num_shards; shard += blockDim.x) {
      i32* owner_completion = params.direct_batch_statuses == nullptr ? nullptr :
        params.direct_batch_statuses +
          static_cast<size_t>(descriptor.query_slot) * params.num_shards + shard;
      shard_status[shard] = direct_fetch_batch(
          params, shard, request_shards, request_offsets, candidate_count,
          params.exact_records + static_cast<size_t>(descriptor.query_slot) *
            params.exact_width * params.node_record_bytes,
          params.node_record_bytes, params.node_record_bytes,
          (descriptor.query_slot + shard) % params.direct_qps_per_node,
          request_local_iova_offsets, owner_completion, true);
    }
    __syncthreads();
    for (u32 shard = threadIdx.x; shard < params.num_shards; shard += blockDim.x) {
      if (shard_status[shard] != -EINPROGRESS) continue;
      i32* owner_completion = params.direct_batch_statuses +
        static_cast<size_t>(descriptor.query_slot) * params.num_shards + shard;
      shard_status[shard] = wait_direct_batch(params, owner_completion);
    }
    __syncthreads();
    for (u32 index = threadIdx.x; index < candidate_count; index += blockDim.x) {
      const u32 delta_slot = request_delta_slots[index];
      if (delta_slot != UINT32_MAX) {
        if (delta_slot < min(load_cg(params.delta_count), params.delta_capacity) &&
            delta_code_visible(params.delta_records[delta_slot],
                               descriptor.snapshot_epoch)) {
          candidate_ids[index] = params.delta_records[delta_slot].id;
          candidate_distances[index] = exact_storage_distance(
            params, query,
            params.delta_vectors +
              static_cast<size_t>(delta_slot) * params.vector_bytes);
        }
        continue;
      }
      const u32 shard = request_shards[index];
      const u32 cache_slot = request_cache_slots[index];
      const bool cache_owned = request_cache_owned[index] != 0;
      if (shard != UINT32_MAX && shard_status[shard] != 0) {
        if (cache_owned) {
          __threadfence();
          atomicExch(params.exact_cache_states + cache_slot, 0u);
        }
        continue;
      }
      if (shard == UINT32_MAX && cache_slot == UINT32_MAX) continue;
      const u8* record = cache_slot != UINT32_MAX
        ? params.exact_cache + static_cast<size_t>(cache_slot) * params.exact_cache_stride
        : params.exact_records +
            (static_cast<size_t>(descriptor.query_slot) * params.exact_width + index) *
              params.node_record_bytes;
      if (exact_record_visible(record)) {
        candidate_ids[index] =
          *reinterpret_cast<const u32*>(record + kNodeIdOffset);
        candidate_distances[index] = exact_storage_distance(
          params, query, record + kNodeVectorOffset);
      }
      if (shard != UINT32_MAX) atomicAdd(exact_reads, 1u);
      if (cache_owned) {
        __threadfence();
        atomicExch(params.exact_cache_states + cache_slot, 2u);
      }
    }
  __syncthreads();
  const u32 existing_count = reset_beam ? 0 : beam_count;
  const u32 merge_count = existing_count + candidate_count;
  for (u32 index = threadIdx.x; index < existing_count; index += blockDim.x) {
    merge_handles[index] = beam_handles[index];
    merge_ids[index] = beam_ids[index];
    merge_distances[index] = beam_distances[index];
    merge_expanded[index] = beam_expanded[index];
  }
  for (u32 index = threadIdx.x; index < candidate_count; index += blockDim.x) {
    const u32 destination = existing_count + index;
    merge_handles[destination] = candidate_handles[index];
    merge_ids[destination] = candidate_ids[index];
    merge_distances[destination] = candidate_distances[index];
    merge_expanded[destination] = 0;
  }
  __syncthreads();
  sort_candidates(merge_handles, merge_ids, merge_distances, merge_expanded,
                  merge_count);
  if (threadIdx.x == 0) {
    u32 valid = 0;
    while (valid < merge_count && merge_handles[valid] != UINT32_MAX &&
           isfinite(merge_distances[valid]) && merge_distances[valid] != FLT_MAX) {
      ++valid;
    }
    beam_count = min(valid, beam_capacity);
  }
  __syncthreads();
  for (u32 index = threadIdx.x; index < beam_count; index += blockDim.x) {
    beam_handles[index] = merge_handles[index];
    beam_ids[index] = merge_ids[index];
    beam_distances[index] = merge_distances[index];
    beam_expanded[index] = merge_expanded[index];
  }
  __syncthreads();
}

__device__ u16 graph_checksum(const u8* data, u32 bytes) {
  u32 hash = 2166136261u;
  for (u32 index = 0; index < bytes; ++index) {
    if (index == 2 || index == 3) continue;
    hash ^= data[index];
    hash *= 16777619u;
  }
  hash ^= hash >> 16;
  return static_cast<u16>(hash);
}

__device__ bool valid_graph_record(const PersistentKernelParams& params, const u8* record) {
  const u16 stored = static_cast<u16>(record[2]) |
    static_cast<u16>(static_cast<u16>(record[3]) << 8);
  return record[0] <= params.graph_degree &&
    stored == graph_checksum(record, params.graph_entry_bytes);
}

__device__ bool prepare_graph_record(const PersistentKernelParams& params,
                                     u32 handle,
                                     u32 query_slot,
                                     u32 request_index,
                                     u32& acquired_slot,
                                     u32& request_shard,
                                     u64& request_offset,
                                     u64& request_local_iova,
                                     bool& cache_hit,
                                     bool& route_hit) {
  acquired_slot = UINT32_MAX;
  request_shard = UINT32_MAX;
  request_offset = 0;
  request_local_iova = 0;
  cache_hit = false;
  route_hit = false;
  u64 raw = 0;
  u64 graph_offset = 0;
  u32 shard = 0;
  if (!resolve_handle(params, handle, raw, shard, graph_offset)) return false;
  const u64 graph_key = (static_cast<u64>(shard) << 48) | graph_offset;
  const u32 route_slot = anchor_graph_slot(params, graph_key);
  if (route_slot != UINT32_MAX && params.anchor_graph_states != nullptr &&
      params.anchor_graph_readers != nullptr &&
      load_cg(params.anchor_graph_states + route_slot) == kGraphCacheReady) {
    atomicAdd(params.anchor_graph_readers + route_slot, 1u);
    __threadfence();
    if (load_cg(params.anchor_graph_states + route_slot) == kGraphCacheReady &&
        load_cg(params.anchor_graph_keys + route_slot) == graph_key) {
      acquired_slot = kGraphRouteBit | route_slot;
      route_hit = true;
      return true;
    }
    atomicSub(params.anchor_graph_readers + route_slot, 1u);
  }
  const u64 generation = load_cg(params.graph_cache_generation);
  const u32 set = params.graph_cache_sets == 0
    ? 0 : hash64(graph_key) % params.graph_cache_sets;
  const u32 way_count = params.graph_cache_ways;

  bool contended = false;
  for (u32 lookup_round = 0;
       lookup_round < 2 && params.graph_cache_sets != 0 && way_count != 0;
       ++lookup_round) {
    bool retry_lookup = false;
    for (u32 way = 0; way < way_count; ++way) {
      const u32 slot = set * way_count + way;
      const u32 state = *reinterpret_cast<volatile u32*>(params.graph_cache_states + slot);
      if (state == kGraphCacheReady &&
          load_cg(params.graph_cache_keys + slot) == graph_key &&
          load_cg(params.graph_cache_generations + slot) == generation) {
        atomicAdd(params.graph_cache_readers + slot, 1u);
        __threadfence();
        const u64 timestamp = load_cg(params.graph_cache_timestamps + slot);
        const u64 now = global_time_ns();
        if (*reinterpret_cast<volatile u32*>(params.graph_cache_states + slot) ==
              kGraphCacheReady &&
            load_cg(params.graph_cache_keys + slot) == graph_key &&
            load_cg(params.graph_cache_generations + slot) == generation &&
            (params.graph_cache_ttl_ns == 0 || now - timestamp <= params.graph_cache_ttl_ns)) {
          acquired_slot = slot;
          cache_hit = true;
          return true;
        }
        atomicSub(params.graph_cache_readers + slot, 1u);
      }
      if ((state == kGraphCacheFilling || state == kGraphCacheFillInvalidated) &&
          load_cg(params.graph_cache_keys + slot) == graph_key &&
          load_cg(params.graph_cache_generations + slot) == generation) {
        u32 wait = 0;
        for (; wait < kCacheWaitRounds; ++wait) {
          const u32 current = *reinterpret_cast<volatile u32*>(
            params.graph_cache_states + slot);
          if ((current != kGraphCacheFilling &&
               current != kGraphCacheFillInvalidated) ||
              *reinterpret_cast<volatile u32*>(params.stop) != 0) break;
          device_ring_relax(128);
        }
        const u32 current = *reinterpret_cast<volatile u32*>(
          params.graph_cache_states + slot);
        retry_lookup = current != kGraphCacheFilling &&
          current != kGraphCacheFillInvalidated;
        contended = !retry_lookup;
        break;
      }
    }
    if (retry_lookup) continue;
    break;
  }

  if (!contended && params.graph_cache_sets != 0 && way_count != 0 &&
      admit_graph_cache(params, graph_key)) {
    const u32 start_way = atomicAdd(params.graph_cache_victims + set, 1u) % way_count;
    for (u32 attempt = 0; attempt < way_count; ++attempt) {
      const u32 slot = set * way_count + (start_way + attempt) % way_count;
      u32 state = *reinterpret_cast<volatile u32*>(params.graph_cache_states + slot);
      if (state == kGraphCacheFilling || state == kGraphCacheFillInvalidated ||
          atomicCAS(params.graph_cache_states + slot, state,
                    kGraphCacheFilling) != state) continue;
      u32 wait = 0;
      while (*reinterpret_cast<volatile u32*>(params.graph_cache_readers + slot) != 0 &&
             *reinterpret_cast<volatile u32*>(params.stop) == 0 &&
             wait++ < kCacheWaitRounds) {
        device_ring_relax(128);
      }
      if (*reinterpret_cast<volatile u32*>(params.stop) != 0) {
        atomicExch(params.graph_cache_states + slot, kGraphCacheEmpty);
        return false;
      }
      if (*reinterpret_cast<volatile u32*>(params.graph_cache_readers + slot) != 0) {
        const u32 current = *reinterpret_cast<volatile u32*>(
          params.graph_cache_states + slot);
        if (current == kGraphCacheFilling) {
          atomicCAS(params.graph_cache_states + slot, kGraphCacheFilling, state);
        } else if (current == kGraphCacheFillInvalidated) {
          atomicCAS(params.graph_cache_states + slot,
                    kGraphCacheFillInvalidated, kGraphCacheStale);
        }
        continue;
      }
      params.graph_cache_keys[slot] = graph_key;
      params.graph_cache_generations[slot] = generation;
      __threadfence();
      u8* destination = params.graph_cache +
        static_cast<size_t>(slot) * kPersistentGraphCacheLineBytes;
      acquired_slot = slot;
      request_shard = shard;
      request_offset = graph_offset;
      request_local_iova = reinterpret_cast<u64>(destination) -
        params.direct_local_iova_base;
      return true;
    }
  }

  if (*reinterpret_cast<volatile u32*>(params.stop) != 0 ||
      params.graph_scratch == nullptr || request_index >= kPersistentMaxPrefetch) {
    return false;
  }
  u8* destination = params.graph_scratch +
    (static_cast<size_t>(query_slot) * kPersistentMaxPrefetch + request_index) *
      kPersistentGraphCacheLineBytes;
  acquired_slot = kGraphScratchBit | request_index;
  request_shard = shard;
  request_offset = graph_offset;
  request_local_iova = reinterpret_cast<u64>(destination) -
    params.direct_local_iova_base;
  return true;
}

__device__ u8* graph_record_pointer(const PersistentKernelParams& params,
                                    u32 query_slot, u32 acquired_slot) {
  if ((acquired_slot & kGraphRouteBit) != 0) {
    const u32 route_slot = acquired_slot & kGraphSlotMask;
    return const_cast<u8*>(params.anchor_graph_records) +
      static_cast<size_t>(route_slot) * params.graph_entry_bytes;
  }
  if ((acquired_slot & kGraphScratchBit) != 0) {
    const u32 request_index = acquired_slot & kGraphSlotMask;
    return params.graph_scratch +
      (static_cast<size_t>(query_slot) * kPersistentMaxPrefetch + request_index) *
        kPersistentGraphCacheLineBytes;
  }
  return params.graph_cache +
    static_cast<size_t>(acquired_slot) * kPersistentGraphCacheLineBytes;
}

__device__ bool fetch_graph_records_batch(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    const u32* handles,
    u32 count,
    u32* acquired_slots,
    u32* remote_reads,
    u32* cache_hits,
    u32* route_hits,
    u32* remote_batches) {
  __shared__ i32 shard_status[kPersistentMaxShards];
  __shared__ u32 failed;
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) * kPersistentMaxMergeCandidates;
  u32* request_shards = params.dynamic_code_request_shards + request_base;
  u64* request_offsets = params.dynamic_code_request_offsets + request_base;
  u64* request_local_iovas =
    params.dynamic_code_request_local_iovas + request_base;

  if (threadIdx.x == 0) failed = 0;
  for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
    acquired_slots[index] = UINT32_MAX;
    request_shards[index] = UINT32_MAX;
    request_offsets[index] = 0;
    request_local_iovas[index] = 0;
    remote_reads[index] = 0;
    cache_hits[index] = 0;
    route_hits[index] = 0;
  }
  for (u32 shard = threadIdx.x; shard < params.num_shards; shard += blockDim.x) {
    shard_status[shard] = 0;
  }
  __syncthreads();

  constexpr u32 warp_width = 32;
  const u32 warp = threadIdx.x / warp_width;
  const u32 lane_in_warp = threadIdx.x % warp_width;
  const u32 warp_count = max(1u, blockDim.x / warp_width);
  if (lane_in_warp == 0) {
    for (u32 index = warp; index < count; index += warp_count) {
      bool cache_hit = false;
      bool route_hit = false;
      if (!prepare_graph_record(params, handles[index], descriptor.query_slot,
                                index, acquired_slots[index],
                                request_shards[index], request_offsets[index],
                                request_local_iovas[index], cache_hit,
                                route_hit)) {
        atomicExch(&failed, 1u);
      } else if (route_hit) {
        route_hits[index] = 1;
      } else if (cache_hit) {
        cache_hits[index] = 1;
      } else {
        remote_reads[index] = 1;
      }
    }
  }
  __syncthreads();
  if (failed != 0) {
    for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
      const u32 slot = acquired_slots[index];
      if (slot == UINT32_MAX) continue;
      if ((slot & kGraphRouteBit) != 0) {
        release_graph_record(params, slot);
      } else if ((slot & kGraphScratchBit) != 0) {
        acquired_slots[index] = UINT32_MAX;
      } else if (cache_hits[index] != 0) {
        release_graph_record(params, slot);
      } else {
        atomicExch(params.graph_cache_states + slot, kGraphCacheEmpty);
      }
      acquired_slots[index] = UINT32_MAX;
    }
    __syncthreads();
    return false;
  }

  for (u32 shard = threadIdx.x; shard < params.num_shards; shard += blockDim.x) {
    u32 matching = 0;
    for (u32 index = 0; index < count; ++index) {
      matching += request_shards[index] == shard ? 1u : 0u;
    }
    if (matching != 0) atomicAdd(remote_batches, 1u);
    i32* owner_completion = params.direct_batch_statuses == nullptr ? nullptr :
      params.direct_batch_statuses +
        static_cast<size_t>(descriptor.query_slot) * params.num_shards + shard;
    shard_status[shard] = direct_fetch_batch(
        params, shard, request_shards, request_offsets, count,
        params.graph_cache, kPersistentGraphCacheLineBytes,
        params.graph_entry_bytes,
        (descriptor.query_slot + shard) % params.direct_qps_per_node,
        request_local_iovas, owner_completion, true);
  }
  __syncthreads();
  for (u32 shard = threadIdx.x; shard < params.num_shards; shard += blockDim.x) {
    if (shard_status[shard] != -EINPROGRESS) continue;
    i32* owner_completion = params.direct_batch_statuses +
      static_cast<size_t>(descriptor.query_slot) * params.num_shards + shard;
    shard_status[shard] = wait_direct_batch(params, owner_completion);
  }
  __syncthreads();

  for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
    const u32 shard = request_shards[index];
    if (shard == UINT32_MAX) continue;
    const u32 slot = acquired_slots[index];
    const bool scratch = (slot & kGraphScratchBit) != 0;
    u8* record = graph_record_pointer(params, descriptor.query_slot, slot);
    const bool ready = shard_status[shard] == 0 && valid_graph_record(params, record);
    if (ready && !scratch) {
      params.graph_cache_timestamps[slot] = global_time_ns();
      params.graph_cache_readers[slot] = 1;
      __threadfence();
      const u32 state = atomicCAS(params.graph_cache_states + slot,
                                  kGraphCacheFilling, kGraphCacheReady);
      if (state == kGraphCacheFillInvalidated) {
        atomicCAS(params.graph_cache_states + slot,
                  kGraphCacheFillInvalidated, kGraphCacheStale);
      } else if (state != kGraphCacheFilling) {
        atomicExch(&failed, 1u);
      }
    } else if (!ready) {
      __threadfence();
      if (!scratch) atomicExch(params.graph_cache_states + slot, kGraphCacheEmpty);
      acquired_slots[index] = UINT32_MAX;
      atomicExch(&failed, 1u);
      if (shard_status[shard] == 0) {
        if (params.direct_error != nullptr) atomicCAS(params.direct_error, 0, -EBADMSG);
        atomicExch(params.direct_disabled, 1u);
      }
    }
  }
  for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
    if (route_hits[index] == 0) continue;
    const u32 slot = acquired_slots[index];
    const u8* record = graph_record_pointer(
      params, descriptor.query_slot, slot);
    if (!valid_graph_record(params, record)) {
      atomicExch(&failed, 1u);
      if (params.direct_error != nullptr) {
        atomicCAS(params.direct_error, 0, -EBADMSG);
      }
      atomicExch(params.direct_disabled, 1u);
    }
  }
  __syncthreads();
  return failed == 0;
}

}  // namespace gpu_search::persistent_kernel_detail
