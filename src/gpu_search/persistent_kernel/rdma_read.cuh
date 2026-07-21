#pragma once

#include "gpu_search/graph_record_validation.hh"
#include "gpu_search/persistent_kernel/candidate_scoring.cuh"

namespace gpu_search::persistent_kernel_detail {

#ifdef DVSTOR_HAVE_GPUNETIO
__device__ __forceinline__ void record_owner_watchdog_counter(
    unsigned long long* counter) {
  if (counter == nullptr) return;
  atomicAdd(counter, 1ULL);
}
#endif

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
  if (finite_f32_bits(distance) && distance < FLT_MAX) return distance;
  double wide_distance = 0.0;
  for (u32 dimension = 0; dimension < params.dim; ++dimension) {
    const double component = static_cast<double>(
      storage_component(params, vector, dimension));
    const double difference = static_cast<double>(query[dimension]) - component;
    wide_distance += difference * difference;
  }
  return saturate_device_squared_l2(wide_distance);
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
    DirectOwnerProgress* watchdog_progress =
      params.direct_owner_progress == nullptr
        ? nullptr : params.direct_owner_progress + qp_index;
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
    // Announce before attempting the bounded enqueue. This also covers an
    // owner that stopped before dequeueing enough entries to free a ring slot.
    // Cancellation before publication balances the monotonic counters below.
    if (watchdog_progress != nullptr) {
      record_owner_watchdog_counter(&watchdog_progress->announced);
    }
    while (!device_ring_try_push(params.direct_batch_queues[qp_index], descriptor)) {
      if (*reinterpret_cast<const volatile u32*>(params.stop) != 0) {
        atomicExch(owner_completion, -ECANCELED);
        if (watchdog_progress != nullptr) {
          record_owner_watchdog_counter(&watchdog_progress->completed);
        }
        return -ECANCELED;
      }
      if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) != 0) {
        atomicExch(owner_completion, -EHOSTDOWN);
        if (watchdog_progress != nullptr) {
          record_owner_watchdog_counter(&watchdog_progress->completed);
        }
        return -EHOSTDOWN;
      }
      // A full descriptor ring is bounded backpressure, not evidence that the
      // QP has failed.  The owner warp is the sole WQE/CQ authority and its CQ
      // watchdog below is responsible for declaring a transport failure.
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
      // Waiting behind already admitted bounded work is normal under load.
      // Only the owner that posts the WQE may turn a missing/error CQE into a
      // global fail-stop transition.
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
  for (;;) {
    const i32 status = *reinterpret_cast<volatile i32*>(owner_completion);
    if (status != -EINPROGRESS) return status;
    if (*reinterpret_cast<const volatile u32*>(params.stop) != 0) {
      return -ECANCELED;
    }
    if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) != 0) {
      return -EHOSTDOWN;
    }
    // This descriptor is already owned by a bounded owner queue.  Queueing
    // delay must not poison every query; the posting owner enforces the actual
    // CQ timeout and publishes a transport-wide failure when appropriate.
    device_ring_relax(128);
  }
#else
  (void)params;
  (void)owner_completion;
  return -ENOTSUP;
#endif
}

__device__ bool exact_record_visible(const PersistentKernelParams& params,
                                     const u8* record, u64 handle) {
  const u64 before = *reinterpret_cast<const u64*>(record);
  const u64 after = *reinterpret_cast<const u64*>(
    record + params.node_record_bytes);
  const u32 expected_incarnation = remote_incarnation(handle);
  const u32 stored_incarnation = *reinterpret_cast<const u32*>(
    record + params.node_incarnation_offset);
  return before == after &&
    (before & (kNodeLockMask | kNodeDeletedMask)) == 0 &&
    static_cast<u32>(before >> kNodeHeaderIncarnationShift) ==
      expected_incarnation &&
    stored_incarnation == expected_incarnation;
}

__device__ bool approximate_handles_batch(const PersistentKernelParams& params,
                                          const QueryDescriptor& descriptor,
                                          const f32* query_lut,
                                          u64* handles,
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
    const u64 handle = handles[index];
    request_shards[index] = UINT32_MAX;
    request_offsets[index] = 0;
    request_local_iova_offsets[index] = 0;
    distances[index] = FLT_MAX;
    if (handle == kInvalidDeviceHandle) continue;
    u32 static_ordinal = 0;
    if (static_ordinal_from_raw(params, handle, static_ordinal)) {
      if (static_ordinal < params.num_nodes) {
        distances[index] = approximate_entry(
          params, query_lut,
          params.pq_codes +
            static_cast<size_t>(static_ordinal) * params.pq_code_bytes);
      }
      continue;
    }

    u64 raw = 0;
    u64 graph_offset = 0;
    u32 shard = 0;
    if (!resolve_handle(params, handle, raw, shard, graph_offset)) continue;
    // Dynamic PQ is authoritative in the storage record. Centroid-route seeds
    // and discovered neighbors therefore use the same one-sided RDMA path.
    if (params.dynamic_code_records == nullptr || shard >= params.num_shards) continue;
    const u64 node_offset = remote_byte_offset(raw);
    request_shards[index] = shard;
    request_offsets[index] = node_offset + params.shards[shard].dynamic_code_offset;
    u8* destination = params.dynamic_code_records +
      (request_base + index) * params.dynamic_code_record_bytes;
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
        params.dynamic_code_records +
          request_base * params.dynamic_code_record_bytes,
        params.dynamic_code_record_bytes, params.dynamic_code_record_bytes,
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
      (request_base + index) * params.dynamic_code_record_bytes;
    if (*reinterpret_cast<const u32*>(code) == remote_incarnation(handles[index])) {
      distances[index] = approximate_entry(
        params, query_lut,
        code + sizeof(u32));
    }
  }
  __syncthreads();
  return failed == 0;
}

__device__ void exactify_into_beam(const PersistentKernelParams& params,
                                   const QueryDescriptor& descriptor,
                                   const f32* query, u64* candidate_handles,
                                   u32* candidate_ids, f32* candidate_distances,
                                   u32 candidate_count, u64* beam_handles,
                                   u32* beam_ids, f32* beam_distances,
                                   u8* beam_expanded, u32& beam_count,
                                   u32* exact_reads,
                                   u32 beam_capacity, bool reset_beam,
                                   u64* merge_handles, u32* merge_ids,
                                   f32* merge_distances, u8* merge_expanded) {
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
      candidate_ids[index] = UINT32_MAX;
      candidate_distances[index] = FLT_MAX;
      const u64 handle = candidate_handles[index];
      u64 raw = 0;
      u64 graph_offset = 0;
      u32 shard = 0;
      if (!resolve_handle(params, handle, raw, shard, graph_offset)) continue;
      request_offsets[index] = remote_byte_offset(raw) +
        params.node_meta_offset;
      request_shards[index] = shard;
      const u8* destination = params.exact_records +
        (static_cast<size_t>(descriptor.query_slot) * params.exact_width + index) *
          params.node_record_stride;
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
            params.exact_width * params.node_record_stride,
          params.node_record_stride, params.node_record_bytes,
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

    // A second header read closes overwrite/flag-update TOCTOU around the
    // first full record fetch. Store it in the validation trailer of each
    // exact-record stride.
    for (u32 index = threadIdx.x; index < candidate_count;
         index += blockDim.x) {
      if (request_shards[index] == UINT32_MAX) continue;
      if (shard_status[request_shards[index]] != 0) {
        request_shards[index] = UINT32_MAX;
        continue;
      }
      u8* destination = params.exact_records +
        (static_cast<size_t>(descriptor.query_slot) * params.exact_width +
         index) * params.node_record_stride + params.node_record_bytes;
      request_local_iova_offsets[index] =
        reinterpret_cast<u64>(destination) - params.direct_local_iova_base;
    }
    for (u32 shard = threadIdx.x; shard < params.num_shards;
         shard += blockDim.x) {
      shard_status[shard] = 0;
    }
    __syncthreads();
    for (u32 shard = threadIdx.x; shard < params.num_shards;
         shard += blockDim.x) {
      i32* owner_completion = params.direct_batch_statuses == nullptr
        ? nullptr : params.direct_batch_statuses +
          static_cast<size_t>(descriptor.query_slot) * params.num_shards +
          shard;
      shard_status[shard] = direct_fetch_batch(
        params, shard, request_shards, request_offsets, candidate_count,
        params.exact_records + static_cast<size_t>(descriptor.query_slot) *
          params.exact_width * params.node_record_stride +
          params.node_record_bytes,
        params.node_record_stride, sizeof(u64),
        (descriptor.query_slot + shard) % params.direct_qps_per_node,
        request_local_iova_offsets, owner_completion, true);
    }
    __syncthreads();
    for (u32 shard = threadIdx.x; shard < params.num_shards;
         shard += blockDim.x) {
      if (shard_status[shard] != -EINPROGRESS) continue;
      i32* owner_completion = params.direct_batch_statuses +
        static_cast<size_t>(descriptor.query_slot) * params.num_shards + shard;
      shard_status[shard] = wait_direct_batch(params, owner_completion);
    }
    __syncthreads();
    for (u32 index = threadIdx.x; index < candidate_count; index += blockDim.x) {
      const u32 shard = request_shards[index];
      if (shard != UINT32_MAX && shard_status[shard] != 0) {
        continue;
      }
      if (shard == UINT32_MAX) continue;
      const u8* record = params.exact_records +
        (static_cast<size_t>(descriptor.query_slot) * params.exact_width + index) *
          params.node_record_stride;
      if (exact_record_visible(params, record, candidate_handles[index])) {
        candidate_ids[index] =
          *reinterpret_cast<const u32*>(record + kNodeIdOffset);
        candidate_distances[index] = exact_storage_distance(
          params, query, record + params.node_vector_offset);
      }
      if (shard != UINT32_MAX) atomicAdd(exact_reads, 1u);
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
    while (valid < merge_count &&
           merge_handles[valid] != kInvalidDeviceHandle &&
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

__device__ graph_record_validation::SnapshotState classify_graph_record(
    const PersistentKernelParams& params, const u8* record, u64 handle) {
  return graph_record_validation::classify_snapshot(
    record, params.graph_entry_bytes, params.graph_degree,
    params.graph_entry_capacity, remote_incarnation(handle));
}

__device__ bool prepare_graph_record(const PersistentKernelParams& params,
                                     u64 handle,
                                     u32 query_slot,
                                     u32 request_index,
                                     u32& acquired_slot,
                                     u32& request_shard,
                                     u64& request_offset,
                                     u64& request_local_iova) {
  acquired_slot = UINT32_MAX;
  request_shard = UINT32_MAX;
  request_offset = 0;
  request_local_iova = 0;
  u64 raw = 0;
  u64 graph_offset = 0;
  u32 shard = 0;
  if (!resolve_handle(params, handle, raw, shard, graph_offset)) return false;
  // Graph records are mutable and versioned by their storage checksum. Always
  // fetch the authoritative record so provisional backlinks and tombstones
  // need no query-side invalidation overlay.
  if (*reinterpret_cast<volatile u32*>(params.stop) != 0 ||
      params.graph_scratch == nullptr || request_index >= kPersistentMaxPrefetch) {
    return false;
  }
  u8* destination = params.graph_scratch +
    (static_cast<size_t>(query_slot) * kPersistentMaxPrefetch + request_index) *
      kPersistentGraphReadBytes;
  acquired_slot = kGraphScratchBit | request_index;
  request_shard = shard;
  request_offset = graph_offset;
  request_local_iova = reinterpret_cast<u64>(destination) -
    params.direct_local_iova_base;
  return true;
}

__device__ u8* graph_record_pointer(const PersistentKernelParams& params,
                                    u32 query_slot, u32 acquired_slot) {
  if ((acquired_slot & kGraphScratchBit) != 0) {
    const u32 request_index = acquired_slot & ~kGraphScratchBit;
    return params.graph_scratch +
      (static_cast<size_t>(query_slot) * kPersistentMaxPrefetch + request_index) *
        kPersistentGraphReadBytes;
  }
  return nullptr;
}

__device__ bool fetch_graph_records_batch(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    const u64* handles,
    u32 count,
    u32* acquired_slots,
    u32* remote_reads,
    u32* remote_batches,
    u32* graph_read_retries) {
  __shared__ i32 shard_status[kPersistentMaxShards];
  __shared__ u32 failed;
  __shared__ u32 retry_pending;
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
      if (!prepare_graph_record(params, handles[index], descriptor.query_slot,
                                index, acquired_slots[index],
                                request_shards[index], request_offsets[index],
                                request_local_iovas[index])) {
        atomicExch(&failed, 1u);
      } else {
        remote_reads[index] = 1;
      }
    }
  }
  __syncthreads();
  if (failed != 0) {
    for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
      acquired_slots[index] = UINT32_MAX;
    }
    __syncthreads();
    return false;
  }

  // A compact graph entry is updated in-place by stage2/reverse-edge workers.
  // Its checksum is therefore an optimistic snapshot validator: a successful
  // RDMA completion can still overlap a legal publication and contain a torn
  // mix of the old and new entry. Storage CPU readers already retry this same
  // condition. Re-read only invalid entries and reserve fail-stop for a record
  // that remains invalid after the bounded snapshot attempts.
  constexpr u32 kGraphSnapshotAttempts = 3;
  for (u32 attempt = 0; attempt < kGraphSnapshotAttempts; ++attempt) {
    if (threadIdx.x == 0) retry_pending = 0;
    for (u32 shard = threadIdx.x; shard < params.num_shards;
         shard += blockDim.x) {
      shard_status[shard] = 0;
    }
    __syncthreads();

    for (u32 shard = threadIdx.x; shard < params.num_shards;
         shard += blockDim.x) {
      u32 matching = 0;
      for (u32 index = 0; index < count; ++index) {
        matching += request_shards[index] == shard ? 1u : 0u;
      }
      if (matching != 0) {
        atomicAdd(remote_batches, 1u);
        if (attempt != 0 && graph_read_retries != nullptr) {
          atomicAdd(graph_read_retries, matching);
        }
      }
      i32* owner_completion = params.direct_batch_statuses == nullptr ? nullptr :
        params.direct_batch_statuses +
          static_cast<size_t>(descriptor.query_slot) * params.num_shards + shard;
      shard_status[shard] = direct_fetch_batch(
          params, shard, request_shards, request_offsets, count,
          params.graph_scratch, kPersistentGraphReadBytes,
          params.graph_entry_bytes,
          (descriptor.query_slot + shard) % params.direct_qps_per_node,
          request_local_iovas, owner_completion, true);
    }
    __syncthreads();
    for (u32 shard = threadIdx.x; shard < params.num_shards;
         shard += blockDim.x) {
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
      u8* record = graph_record_pointer(params, descriptor.query_slot, slot);
      const i32 status = shard_status[shard];
      const graph_record_validation::SnapshotState snapshot = status == 0
        ? classify_graph_record(params, record, handles[index])
        : graph_record_validation::SnapshotState::invalid;
      const graph_record_validation::ReadAction action =
        graph_record_validation::decide_read_action(
          status == 0, snapshot, attempt + 1 < kGraphSnapshotAttempts);
      if (action == graph_record_validation::ReadAction::accept) {
        // UINT32_MAX removes this entry from subsequent per-shard retry
        // batches while leaving acquired_slots intact for traversal.
        request_shards[index] = UINT32_MAX;
        continue;
      }
      if (action == graph_record_validation::ReadAction::discard_stale) {
        // Slot reuse is an ordinary read-committed race.  The tagged handle no
        // longer names this record, so drop only this expansion; disabling the
        // direct path would turn normal update churn into a service failure.
        acquired_slots[index] = UINT32_MAX;
        request_shards[index] = UINT32_MAX;
        continue;
      }
      if (action == graph_record_validation::ReadAction::retry) {
        atomicAdd(&retry_pending, 1u);
        continue;
      }

      acquired_slots[index] = UINT32_MAX;
      request_shards[index] = UINT32_MAX;
      atomicExch(&failed, 1u);
      if (status == 0) {
        if (params.direct_error != nullptr) {
          atomicCAS(params.direct_error, 0, -EBADMSG);
        }
        atomicExch(params.direct_disabled, 1u);
      }
    }
    __syncthreads();
    if (retry_pending == 0) break;
    if (threadIdx.x == 0) device_ring_relax(128);
    __syncthreads();
  }
  return failed == 0;
}

}  // namespace gpu_search::persistent_kernel_detail
