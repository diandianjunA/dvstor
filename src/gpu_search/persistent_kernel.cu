#include "gpu_search/persistent_kernel.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cerrno>
#include <cmath>
#include <cstdint>

#ifdef DVSTOR_HAVE_GPUNETIO
#ifndef IBV_WC_DRIVER1
#define IBV_WC_DRIVER1 135
#define IBV_WC_DRIVER2 136
#define IBV_WC_DRIVER3 137
#endif
#include <doca_gpunetio_dev_verbs_onesided.cuh>
#endif

namespace gpu_search {
namespace {

__device__ __forceinline__ u32 hash32(u32 value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

__device__ __forceinline__ u32 hash64(u64 value) {
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdULL;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ULL;
  value ^= value >> 33;
  return static_cast<u32>(value ^ (value >> 32));
}

__device__ __forceinline__ u64 load_cg(const u64* address) {
  u64 value = 0;
  asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(value) : "l"(address));
  return value;
}

__device__ __forceinline__ u32 load_cg(const u32* address) {
  u32 value = 0;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(address));
  return value;
}

__device__ __forceinline__ u64 global_time_ns() {
  u64 value = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

#ifdef DVSTOR_HAVE_GPUNETIO
__device__ i32 poll_direct_cq(doca_gpu_dev_verbs_cq* completion_queue,
                              doca_gpu_dev_verbs_ticket_t ticket,
                              u64 timeout_ns, const u32* stop,
                              const u32* direct_disabled) {
  auto* completion_base = reinterpret_cast<mlx5_cqe64*>(
    __ldg(reinterpret_cast<uintptr_t*>(&completion_queue->cqe_daddr)));
  const u32 completion_count = __ldg(&completion_queue->cqe_num);
  const u32 completion_index = ticket & (completion_count - 1);
  auto* completion = completion_base + completion_index;
  const u64 started = global_time_ns();
  for (;;) {
    const u64 consumer =
      doca_gpu_dev_verbs_load_relaxed<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
        &completion_queue->cqe_ci);
    const u8 owner = doca_gpu_dev_verbs_load_relaxed_sys_global(
      reinterpret_cast<u8*>(&completion->op_own));
    if (!((consumer <= ticket) &&
          ((owner & MLX5_CQE_OWNER_MASK) ^ !!(ticket & completion_count)))) {
      const u8 opcode = owner >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;
      const i32 status = opcode == MLX5_CQE_REQ_ERR ? -EIO : 0;
      if (status == 0) {
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
        doca_gpu_dev_verbs_atomic_max<
          u64, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
            &completion_queue->cqe_ci, ticket + 1);
      }
      return status;
    }
    if (*reinterpret_cast<const volatile u32*>(stop) != 0 ||
        *reinterpret_cast<const volatile u32*>(direct_disabled) != 0) {
      return -ECANCELED;
    }
    if (global_time_ns() - started >= timeout_ns) return -ETIMEDOUT;
    device_ring_relax(128);
  }
}

__device__ i32 lock_direct_qp(i32* lock, const u32* stop,
                              const u32* direct_disabled) {
  if (*reinterpret_cast<const volatile u32*>(stop) != 0 ||
      *reinterpret_cast<const volatile u32*>(direct_disabled) != 0) {
    return -ECANCELED;
  }
  while (atomicCAS(lock, 0, 1) != 0) {
    if (*reinterpret_cast<const volatile u32*>(stop) != 0 ||
        *reinterpret_cast<const volatile u32*>(direct_disabled) != 0) {
      return -ECANCELED;
    }
    device_ring_relax(128);
  }
  if (*reinterpret_cast<const volatile u32*>(stop) != 0 ||
      *reinterpret_cast<const volatile u32*>(direct_disabled) != 0) {
    atomicExch(lock, 0);
    return -ECANCELED;
  }
  __threadfence();
  return 0;
}

__device__ void unlock_direct_qp(i32* lock) {
  __threadfence();
  atomicExch(lock, 0);
}
#endif

__device__ bool insert_visited(u32* table, u32 capacity, u32 handle) {
  const u32 mask = capacity - 1;
  u32 slot = hash32(handle) & mask;
  for (u32 probe = 0; probe < capacity; ++probe) {
    const u32 old = atomicCAS(table + slot, UINT32_MAX, handle);
    if (old == UINT32_MAX) return true;
    if (old == handle) return false;
    slot = (slot + 1) & mask;
  }
  return false;
}

__device__ const DeviceShardRegion* shard_for_ordinal(
    const PersistentKernelParams& params, u32 ordinal, u64* slot_out = nullptr) {
  for (u32 shard = 0; shard < params.num_shards; ++shard) {
    const DeviceShardRegion& region = params.shards[shard];
    if (ordinal >= region.ordinal_base &&
        static_cast<u64>(ordinal) - region.ordinal_base < region.node_count) {
      if (slot_out != nullptr) *slot_out = static_cast<u64>(ordinal) - region.ordinal_base;
      return params.shards + shard;
    }
  }
  return nullptr;
}

__device__ bool static_handle_from_raw(const PersistentKernelParams& params,
                                       u64 raw, u32& handle) {
  const u32 shard = static_cast<u32>(raw >> 48);
  const u64 offset = (raw << 16) >> 16;
  if (shard >= params.num_shards) return false;
  const DeviceShardRegion& region = params.shards[shard];
  if (offset < region.node_base_offset || region.node_stride == 0) return false;
  const u64 relative = offset - region.node_base_offset;
  if (relative % region.node_stride != 0) return false;
  const u64 slot = relative / region.node_stride;
  if (slot >= region.node_count) return false;
  handle = static_cast<u32>(region.ordinal_base + slot);
  return true;
}

__device__ u32 delta_slot_from_raw(const PersistentKernelParams& params, u64 raw) {
  if (raw == 0 || params.delta_remote_capacity == 0) return UINT32_MAX;
  const u32 mask = params.delta_remote_capacity - 1;
  u32 position = hash64(raw) & mask;
  for (u32 probe = 0; probe < params.delta_remote_capacity; ++probe) {
    const u64 key = load_cg(params.delta_remote_keys + position);
    if (key == raw) return load_cg(params.delta_remote_slots + position);
    if (key == 0) return UINT32_MAX;
    position = (position + 1) & mask;
  }
  return UINT32_MAX;
}

__device__ u32 handle_from_raw(const PersistentKernelParams& params, u64 raw) {
  u32 handle = UINT32_MAX;
  if (static_handle_from_raw(params, raw, handle)) return handle;
  const u32 slot = delta_slot_from_raw(params, raw);
  return slot == UINT32_MAX ? UINT32_MAX : kDeltaHandleBit | slot;
}

__device__ bool resolve_handle(const PersistentKernelParams& params, u32 handle,
                               u64& raw, u32& shard, u64& graph_offset) {
  if ((handle & kDeltaHandleBit) == 0) {
    u64 slot = 0;
    const DeviceShardRegion* region = shard_for_ordinal(params, handle, &slot);
    if (region == nullptr) return false;
    shard = region->memory_node;
    const u64 node_offset = region->node_base_offset + slot * region->node_stride;
    raw = (static_cast<u64>(shard) << 48) | node_offset;
    graph_offset = region->graph_base_offset + slot * params.graph_entry_bytes;
    return true;
  }
  const u32 delta_slot = handle & kDeltaHandleMask;
  if (delta_slot >= min(load_cg(params.delta_count), params.delta_capacity)) return false;
  raw = load_cg(&params.delta_records[delta_slot].remote_node);
  if (raw == 0) return false;
  shard = static_cast<u32>(raw >> 48);
  const u64 node_offset = (raw << 16) >> 16;
  if (shard >= params.num_shards) return false;
  const DeviceShardRegion& region = params.shards[shard];
  if (node_offset >= region.node_base_offset) {
    const u64 relative = node_offset - region.node_base_offset;
    if (region.node_stride != 0 && relative % region.node_stride == 0 &&
        relative / region.node_stride < region.node_count) {
      graph_offset = region.graph_base_offset +
        (relative / region.node_stride) * params.graph_entry_bytes;
      return true;
    }
  }
  if (node_offset < region.dynamic_base_offset || region.dynamic_record_bytes == 0 ||
      (node_offset - region.dynamic_base_offset) % region.dynamic_record_bytes != 0) {
    return false;
  }
  graph_offset = node_offset + region.dynamic_hot_offset;
  return true;
}

__device__ u64 base_override_epoch(const PersistentKernelParams& params, u32 ordinal) {
  if (params.base_override_capacity == 0) return 0;
  const u32 mask = params.base_override_capacity - 1;
  u32 position = hash32(ordinal) & mask;
  for (u32 probe = 0; probe < params.base_override_capacity; ++probe) {
    const u32 key = load_cg(params.base_override_keys + position);
    if (key == ordinal) return load_cg(params.base_override_epochs + position);
    if (key == UINT32_MAX) return 0;
    position = (position + 1) & mask;
  }
  return 0;
}

__device__ bool delta_visible(const DeviceDeltaRecord& record, u64 snapshot_epoch) {
  const u64 superseded = load_cg(&record.superseded_epoch);
  return record.epoch <= snapshot_epoch &&
    (superseded == 0 || superseded > snapshot_epoch) &&
    (record.flags & kDeltaDeleted) == 0;
}

__device__ f32 approximate_entry(const PersistentKernelParams& params,
                                 const f32* query_lut,
                                 const u8* code) {
  f32 distance = 0.0f;
  for (u32 subquantizer = 0; subquantizer < params.pq_subquantizers;
       ++subquantizer) {
    distance += query_lut[static_cast<size_t>(subquantizer) * 256 +
                          code[subquantizer]];
  }
  return distance;
}

__device__ f32 approximate_handle(const PersistentKernelParams& params,
                                  const f32* query_lut,
                                  u32 handle, u64 snapshot_epoch) {
  if ((handle & kDeltaHandleBit) == 0) {
    if (handle >= params.num_nodes) return FLT_MAX;
    const u64 override_epoch = base_override_epoch(params, handle);
    if (override_epoch != 0 && override_epoch <= snapshot_epoch) return FLT_MAX;
    return approximate_entry(params, query_lut,
      params.pq_codes + static_cast<size_t>(handle) * params.pq_code_bytes);
  }
  const u32 slot = handle & kDeltaHandleMask;
  if (slot >= min(load_cg(params.delta_count), params.delta_capacity)) return FLT_MAX;
  const DeviceDeltaRecord& record = params.delta_records[slot];
  if (!delta_visible(record, snapshot_epoch)) return FLT_MAX;
  return approximate_entry(params, query_lut,
    params.delta_pq_codes + static_cast<size_t>(slot) * params.pq_code_bytes);
}

__device__ void beam_insert(u32* handles, u32* ids, f32* distances, u8* expanded,
                            u32& count, u32 capacity, u32 handle, u32 id, f32 distance) {
  if (handle == UINT32_MAX || !isfinite(distance) || distance == FLT_MAX) return;
  if (count < capacity) {
    handles[count] = handle;
    ids[count] = id;
    distances[count] = distance;
    expanded[count] = 0;
    ++count;
    return;
  }
  u32 worst = 0;
  for (u32 index = 1; index < count; ++index) {
    if (distances[index] > distances[worst]) worst = index;
  }
  if (distance >= distances[worst]) return;
  handles[worst] = handle;
  ids[worst] = id;
  distances[worst] = distance;
  expanded[worst] = 0;
}

__device__ __forceinline__ bool candidate_less(u32 lhs_handle, f32 lhs_distance,
                                               u32 rhs_handle, f32 rhs_distance) {
  return lhs_distance < rhs_distance ||
    (lhs_distance == rhs_distance && lhs_handle < rhs_handle);
}

__device__ u32 candidate_sort_capacity(u32 count) {
  u32 capacity = 1;
  while (capacity < count) capacity <<= 1;
  return capacity;
}

__device__ void sort_candidates(u32* handles, u32* ids, f32* distances,
                                u8* expanded, u32 count) {
  const u32 capacity = candidate_sort_capacity(max(1u, count));
  for (u32 index = count + threadIdx.x; index < capacity; index += blockDim.x) {
    handles[index] = UINT32_MAX;
    if (ids != nullptr) ids[index] = UINT32_MAX;
    distances[index] = FLT_MAX;
    expanded[index] = 0;
  }
  __syncthreads();
  for (u32 sequence = 2; sequence <= capacity; sequence <<= 1) {
    for (u32 stride = sequence >> 1; stride != 0; stride >>= 1) {
      for (u32 index = threadIdx.x; index < capacity; index += blockDim.x) {
        const u32 partner = index ^ stride;
        if (partner <= index) continue;
        const bool ascending = (index & sequence) == 0;
        const bool exchange = ascending
          ? candidate_less(handles[partner], distances[partner],
                           handles[index], distances[index])
          : candidate_less(handles[index], distances[index],
                           handles[partner], distances[partner]);
        if (!exchange) continue;
        const u32 handle = handles[index];
        handles[index] = handles[partner];
        handles[partner] = handle;
        if (ids != nullptr) {
          const u32 id = ids[index];
          ids[index] = ids[partner];
          ids[partner] = id;
        }
        const f32 distance = distances[index];
        distances[index] = distances[partner];
        distances[partner] = distance;
        const u8 was_expanded = expanded[index];
        expanded[index] = expanded[partner];
        expanded[partner] = was_expanded;
      }
      __syncthreads();
    }
  }
}

__device__ void merge_approximate_into_beam(
    u32* candidate_handles, f32* candidate_distances, u32 candidate_count,
    u32* beam_handles, u32* beam_ids, f32* beam_distances,
    u8* beam_expanded, u32& beam_count, u32 beam_capacity,
    u32* merge_handles, u32* merge_ids, f32* merge_distances,
    u8* merge_expanded) {
  const u32 existing_count = beam_count;
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
    merge_ids[destination] = UINT32_MAX;
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

__device__ f32 exact_storage_distance(const PersistentKernelParams& params,
                                      const f32* query, const u8* vector) {
  f32 distance = 0.0f;
  for (u32 dimension = 0; dimension < params.dim; ++dimension) {
    f32 component = 0.0f;
    if (params.vector_dtype == 0) component = reinterpret_cast<const f32*>(vector)[dimension];
    else if (params.vector_dtype == 1) component = static_cast<f32>(vector[dimension]);
    else component = static_cast<f32>(reinterpret_cast<const int8_t*>(vector)[dimension]);
    const f32 difference = query[dimension] - component;
    distance += difference * difference;
  }
  return distance;
}

__device__ f32 exact_float_distance(const PersistentKernelParams& params,
                                    const f32* query, const f32* vector) {
  f32 distance = 0.0f;
  for (u32 dimension = 0; dimension < params.dim; ++dimension) {
    const f32 difference = query[dimension] - vector[dimension];
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
                                  const u64* local_iova_offsets = nullptr) {
#ifdef DVSTOR_HAVE_GPUNETIO
  if (memory_node >= params.direct_region_count || params.direct_qps == nullptr ||
      params.direct_qp_locks == nullptr || params.direct_qps_per_node == 0 ||
      params.direct_disabled == nullptr ||
      *reinterpret_cast<volatile u32*>(params.direct_disabled) != 0) return -EHOSTDOWN;
  u32 matching = 0;
  for (u32 index = 0; index < request_count; ++index) {
    if (request_shards[index] == memory_node) ++matching;
  }
  if (matching == 0) return 0;
  const u32 qp_index = (lane % params.direct_qps_per_node) *
    params.direct_region_count + memory_node;
  if (params.direct_qps[qp_index] == nullptr) return -EINVAL;
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
  return -ENOTSUP;
#endif
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
    constexpr u32 warp_width = 32;
    const u32 warp = threadIdx.x / warp_width;
    const u32 lane_in_warp = threadIdx.x % warp_width;
    const u32 warp_count = max(1u, blockDim.x / warp_width);
    __shared__ u32 request_shards[kPersistentMaxExact];
    __shared__ u64 request_offsets[kPersistentMaxExact];
    __shared__ u64 request_local_iova_offsets[kPersistentMaxExact];
    __shared__ u32 request_cache_slots[kPersistentMaxExact];
    __shared__ u8 request_cache_owned[kPersistentMaxExact];
    __shared__ i32 shard_status[kPersistentMaxShards];
    for (u32 index = threadIdx.x; index < candidate_count; index += blockDim.x) {
      request_shards[index] = UINT32_MAX;
      request_offsets[index] = 0;
      request_cache_slots[index] = UINT32_MAX;
      request_cache_owned[index] = 0;
      candidate_ids[index] = UINT32_MAX;
      candidate_distances[index] = FLT_MAX;
      const u32 handle = candidate_handles[index];
      if ((handle & kDeltaHandleBit) != 0) continue;
      const u64 override_epoch = base_override_epoch(params, handle);
      if (override_epoch != 0 && override_epoch <= descriptor.snapshot_epoch) continue;
      u64 raw = 0;
      u64 graph_offset = 0;
      u32 shard = 0;
      if (!resolve_handle(params, handle, raw, shard, graph_offset)) continue;
      request_offsets[index] = ((raw << 16) >> 16) + params.node_meta_offset;
      u32 cache_slot = UINT32_MAX;
      if (params.exact_cache_sets != 0 && params.exact_cache_ways != 0) {
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
              candidate_ids[index] = *reinterpret_cast<const u32*>(record);
              candidate_distances[index] = exact_storage_distance(params, query, record + 8);
              atomicAdd(exact_cache_hits, 1u);
              cache_hit = true;
            }
            atomicSub(params.exact_cache_readers + slot, 1u);
            if (cache_hit) break;
          }
        }
        if (cache_hit) continue;
        if (cache_slot == UINT32_MAX) {
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
            params.exact_cache_keys[slot] = handle;
            __threadfence();
            while (*reinterpret_cast<volatile u32*>(params.exact_cache_readers + slot) != 0 &&
                   *reinterpret_cast<volatile u32*>(params.stop) == 0) {
              device_ring_relax(128);
            }
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
    for (u32 base = 0; base < params.num_shards; base += warp_count) {
      if (lane_in_warp == 0 && warp < warp_count && base + warp < params.num_shards) {
        const u32 shard = base + warp;
        shard_status[shard] = direct_fetch_batch(
          params, shard, request_shards, request_offsets, candidate_count,
          params.exact_records + static_cast<size_t>(descriptor.query_slot) *
            params.exact_width * params.node_record_bytes,
          params.node_record_bytes, params.node_record_bytes,
          (descriptor.query_slot + shard) % params.direct_qps_per_node,
          request_local_iova_offsets);
      }
      __syncthreads();
    }
    for (u32 index = threadIdx.x; index < candidate_count; index += blockDim.x) {
      const u32 handle = candidate_handles[index];
      if ((handle & kDeltaHandleBit) != 0) {
        const u32 slot = handle & kDeltaHandleMask;
        if (slot < min(load_cg(params.delta_count), params.delta_capacity) &&
            delta_visible(params.delta_records[slot], descriptor.snapshot_epoch)) {
          candidate_ids[index] = params.delta_records[slot].id;
          candidate_distances[index] = exact_float_distance(
            params, query, params.delta_vectors + static_cast<size_t>(slot) * params.dim);
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
      candidate_ids[index] = *reinterpret_cast<const u32*>(record);
      candidate_distances[index] = exact_storage_distance(params, query, record + 8);
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

__device__ const u8* fetch_graph_record(const PersistentKernelParams& params,
                                        const QueryDescriptor& descriptor,
                                        u32 handle, u32 request_lane,
                                        u32& remote_reads,
                                        u32& cache_hits, u32& acquired_slot) {
  acquired_slot = UINT32_MAX;
  u64 raw = 0;
  u64 graph_offset = 0;
  u32 shard = 0;
  if (!resolve_handle(params, handle, raw, shard, graph_offset) ||
      params.graph_cache_sets == 0) return nullptr;
  const u64 graph_key = (static_cast<u64>(shard) << 48) | graph_offset;
  const u64 generation = load_cg(params.graph_cache_generation);
  const u32 set = hash64(graph_key) % params.graph_cache_sets;
  const u32 way_count = params.graph_cache_ways;

  for (;;) {
    bool retry_lookup = false;
    for (u32 way = 0; way < way_count; ++way) {
      const u32 slot = set * way_count + way;
      const u32 state = *reinterpret_cast<volatile u32*>(params.graph_cache_states + slot);
      if (state == 2 && load_cg(params.graph_cache_keys + slot) == graph_key &&
          load_cg(params.graph_cache_generations + slot) == generation) {
        atomicAdd(params.graph_cache_readers + slot, 1u);
        __threadfence();
        const u64 timestamp = load_cg(params.graph_cache_timestamps + slot);
        const u64 now = global_time_ns();
        if (*reinterpret_cast<volatile u32*>(params.graph_cache_states + slot) == 2 &&
            load_cg(params.graph_cache_keys + slot) == graph_key &&
            load_cg(params.graph_cache_generations + slot) == generation &&
            (params.graph_cache_ttl_ns == 0 || now - timestamp <= params.graph_cache_ttl_ns)) {
          ++cache_hits;
          acquired_slot = slot;
          return params.graph_cache +
            static_cast<size_t>(slot) * kPersistentGraphCacheLineBytes;
        }
        atomicSub(params.graph_cache_readers + slot, 1u);
      }
      if (state == 1 && load_cg(params.graph_cache_keys + slot) == graph_key &&
          load_cg(params.graph_cache_generations + slot) == generation) {
        while (*reinterpret_cast<volatile u32*>(params.graph_cache_states + slot) == 1 &&
               *reinterpret_cast<volatile u32*>(params.stop) == 0) {
          device_ring_relax(128);
        }
        retry_lookup = true;
        break;
      }
    }
    if (retry_lookup) continue;

    const u32 start_way = atomicAdd(params.graph_cache_victims + set, 1u) % way_count;
    for (u32 attempt = 0; attempt < way_count; ++attempt) {
      const u32 slot = set * way_count + (start_way + attempt) % way_count;
      u32 state = *reinterpret_cast<volatile u32*>(params.graph_cache_states + slot);
      if (state == 1 || atomicCAS(params.graph_cache_states + slot, state, 1u) != state) continue;
      while (*reinterpret_cast<volatile u32*>(params.graph_cache_readers + slot) != 0 &&
             *reinterpret_cast<volatile u32*>(params.stop) == 0) {
        device_ring_relax(128);
      }
      params.graph_cache_keys[slot] = graph_key;
      params.graph_cache_generations[slot] = generation;
      __threadfence();
      u8* destination = params.graph_cache +
        static_cast<size_t>(slot) * kPersistentGraphCacheLineBytes;
      bool ready = false;
      for (u32 retry = 0; retry < 8 &&
           *reinterpret_cast<volatile u32*>(params.stop) == 0; ++retry) {
        if (*reinterpret_cast<volatile u32*>(params.direct_disabled) != 0) break;
        const i32 status = direct_fetch(params, shard, graph_offset, destination,
                                        params.graph_entry_bytes, request_lane);
        const bool fetched_direct = status == 0;
        if (!fetched_direct) break;
        ++remote_reads;
        ready = valid_graph_record(params, destination);
        if (ready) break;
        if (fetched_direct) {
          if (params.direct_error != nullptr) atomicCAS(params.direct_error, 0, -EBADMSG);
          atomicExch(params.direct_disabled, 1u);
        }
        device_ring_relax(256u << min(retry, 4u));
      }
      if (ready) {
        params.graph_cache_timestamps[slot] = global_time_ns();
        params.graph_cache_readers[slot] = 1;
        __threadfence();
        atomicExch(params.graph_cache_states + slot, 2u);
        acquired_slot = slot;
        return destination;
      }
      __threadfence();
      atomicExch(params.graph_cache_states + slot, 0u);
      return nullptr;
    }
    if (*reinterpret_cast<volatile u32*>(params.stop) != 0) return nullptr;
    device_ring_relax(128);
  }
}

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
                                     u32 beam_capacity) {
  __shared__ u32 delta_count_snapshot;
  if (threadIdx.x == 0) {
    delta_count_snapshot = min(load_cg(params.delta_count), params.delta_capacity);
  }
  __syncthreads();
  const u32 count = delta_count_snapshot;
  if (count == 0) return;
  __shared__ u32 candidate_handles[256];
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
  } else {
    f32* anchor_distances = params.anchor_distances +
      static_cast<size_t>(descriptor.query_slot) * params.anchor_count;
    for (u32 anchor = threadIdx.x; anchor < params.anchor_count; anchor += blockDim.x) {
      anchor_distances[anchor] = exact_float_distance(
        params, query, params.anchor_vectors + static_cast<size_t>(anchor) * params.dim);
    }
    __syncthreads();
    __shared__ u32 selected[kPersistentMaxAnchorProbes];
    __shared__ u32 selected_count;
    __shared__ u32 anchor_best_indices[256];
    __shared__ f32 anchor_best_distances[256];
    if (threadIdx.x == 0) {
      selected_count = min(params.delta_anchor_probes,
                           min(params.anchor_count, kPersistentMaxAnchorProbes));
    }
    __syncthreads();
    for (u32 index = 0; index < selected_count; ++index) {
      u32 local_anchor = UINT32_MAX;
      f32 local_distance = FLT_MAX;
      for (u32 anchor = threadIdx.x; anchor < params.anchor_count;
           anchor += blockDim.x) {
        const f32 distance = anchor_distances[anchor];
        if (distance < local_distance ||
            (distance == local_distance && anchor < local_anchor)) {
          local_distance = distance;
          local_anchor = anchor;
        }
      }
      anchor_best_indices[threadIdx.x] = local_anchor;
      anchor_best_distances[threadIdx.x] = local_distance;
      __syncthreads();
      for (u32 stride = blockDim.x / 2; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride) {
          const f32 candidate_distance = anchor_best_distances[threadIdx.x + stride];
          const u32 candidate_anchor = anchor_best_indices[threadIdx.x + stride];
          if (candidate_distance < anchor_best_distances[threadIdx.x] ||
              (candidate_distance == anchor_best_distances[threadIdx.x] &&
               candidate_anchor < anchor_best_indices[threadIdx.x])) {
            anchor_best_distances[threadIdx.x] = candidate_distance;
            anchor_best_indices[threadIdx.x] = candidate_anchor;
          }
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        selected[index] = anchor_best_indices[0];
        if (selected[index] != UINT32_MAX) {
          anchor_distances[selected[index]] = FLT_MAX;
        }
      }
      __syncthreads();
    }
    if (selected_count != 0) {
      const u32 probe = threadIdx.x % selected_count;
      const u32 partition = threadIdx.x / selected_count;
      const u32 partitions = (blockDim.x - 1 - probe) / selected_count + 1;
      const u32 selected_anchor = selected[probe];
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
  }
  candidate_handles[threadIdx.x] = local_slot == UINT32_MAX
    ? UINT32_MAX : kDeltaHandleBit | local_slot;
  candidate_distances[threadIdx.x] = local_slot == UINT32_MAX
    ? FLT_MAX
    : exact_float_distance(params, query,
        params.delta_vectors + static_cast<size_t>(local_slot) * params.dim);
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
        const u32 slot = handle & kDeltaHandleMask;
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
      descriptor.k == 0 || descriptor.k > descriptor.result_capacity) {
    if (threadIdx.x == 0) {
      completion.status = -EINVAL;
      completion.gpu_cycles = clock64() - query_started_cycles;
      device_ring_push(params.completions, completion);
    }
    __syncthreads();
    return;
  }

  const f32* query = reinterpret_cast<const f32*>(descriptor.query_device_address);
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
  __shared__ u32 merge_handles[kPersistentMaxMergeCandidates];
  __shared__ u32 merge_ids[kPersistentMaxMergeCandidates];
  __shared__ f32 merge_distances[kPersistentMaxMergeCandidates];
  __shared__ u8 merge_expanded[kPersistentMaxMergeCandidates];
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
  for (u32 index = threadIdx.x; index < params.entry_point_count; index += blockDim.x) {
    const u32 handle = params.entry_points[index];
    merge_handles[index] = handle;
    merge_distances[index] = approximate_handle(
      params, query_lut, handle, descriptor.snapshot_epoch);
    merge_expanded[index] = 0;
  }
  __syncthreads();
  sort_candidates(merge_handles, nullptr, merge_distances, merge_expanded,
                  params.entry_point_count);
  if (threadIdx.x == 0) {
    beam_count = min(params.entry_point_count,
                     min(traversal_capacity, params.entry_seed_count));
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
  __shared__ u32 remote_reads_by_lane[kPersistentMaxPrefetch];
  __shared__ u32 cache_hits_by_lane[kPersistentMaxPrefetch];
  __shared__ u32 neighbor_handles[kPersistentMaxPrefetch * kPersistentMaxGraphDegree];
  __shared__ f32 neighbor_distances[kPersistentMaxPrefetch * kPersistentMaxGraphDegree];
  __shared__ u32 total_remote_reads;
  __shared__ u32 total_cache_hits;
  __shared__ u32 graph_failed;
  if (threadIdx.x == 0) {
    total_remote_reads = 0;
    total_cache_hits = 0;
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
      while (selected_count < target) {
        i32 best = -1;
        f32 best_distance = FLT_MAX;
        for (u32 index = 0; index < beam_count; ++index) {
          if (beam_expanded[index] == 0 && beam_distances[index] < best_distance) {
            best = static_cast<i32>(index);
            best_distance = beam_distances[index];
          }
        }
        if (best < 0) break;
        beam_expanded[best] = 1;
        selected_handles[selected_count++] = beam_handles[best];
      }
    }
    __syncthreads();
    if (selected_count == 0) break;
    constexpr u32 warp_width = 32;
    const u32 warp = threadIdx.x / warp_width;
    const u32 lane_in_warp = threadIdx.x % warp_width;
    const u32 graph_parallel_width = max(
      1u, min(min(params.direct_qps_per_node, blockDim.x / warp_width),
              selected_count));
    for (u32 base = 0; base < selected_count; base += graph_parallel_width) {
      const bool direct_worker = lane_in_warp == 0 && warp < graph_parallel_width;
      const u32 worker = warp;
      if (direct_worker && base + worker < selected_count) {
        const u32 lane = base + worker;
        const u32 request_lane =
          (descriptor.query_slot + base + worker) % params.direct_qps_per_node;
        remote_reads_by_lane[lane] = 0;
        cache_hits_by_lane[lane] = 0;
        u32 acquired_slot = UINT32_MAX;
        const u8* record = fetch_graph_record(
          params, descriptor, selected_handles[lane], request_lane,
          remote_reads_by_lane[lane], cache_hits_by_lane[lane],
          acquired_slot);
        u32 count = 0;
        if (record != nullptr) {
          count = (record[1] & 1u) == 0
            ? min(static_cast<u32>(record[0]), params.graph_degree) : 0;
          for (u32 neighbor = 0; neighbor < count; ++neighbor) {
            const u64 raw = decode_compact_raw(record + 8 + neighbor * 5,
                                               params.graph_shard_bits);
            neighbor_handles[lane * kPersistentMaxGraphDegree + neighbor] =
              handle_from_raw(params, raw);
          }
          __threadfence();
          atomicSub(params.graph_cache_readers + acquired_slot, 1u);
        } else {
          atomicExch(&graph_failed, 1u);
        }
        neighbor_counts[lane] = count;
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      graph_phase_cycles += clock64() - phase_started_cycles;
      phase_started_cycles = clock64();
    }
    __syncthreads();
    if (graph_failed != 0) {
      if (threadIdx.x == 0) {
        completion.status = -EIO;
        completion.gpu_cycles = clock64() - query_started_cycles;
        completion.remote_pages = total_remote_reads;
        completion.cache_hits = total_cache_hits;
        completion.exact_vectors = total_exact_reads;
        completion.exact_cache_hits = total_exact_cache_hits;
        device_ring_push(params.completions, completion);
      }
      __syncthreads();
      return;
    }
    u32 flattened = 0;
    for (u32 lane = 0; lane < selected_count; ++lane) flattened += neighbor_counts[lane];
    for (u32 flat = threadIdx.x; flat < flattened; flat += blockDim.x) {
      u32 lane = 0;
      u32 relative = flat;
      while (lane < selected_count && relative >= neighbor_counts[lane]) {
        relative -= neighbor_counts[lane++];
      }
      const u32 offset = lane * kPersistentMaxGraphDegree + relative;
      const u32 handle = neighbor_handles[offset];
      neighbor_distances[offset] = handle != UINT32_MAX &&
          insert_visited(visited, params.visited_capacity, handle)
        ? approximate_handle(params, query_lut, handle,
                             descriptor.snapshot_epoch)
        : FLT_MAX;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      score_phase_cycles += clock64() - phase_started_cycles;
      phase_started_cycles = clock64();
    }
    __syncthreads();
    const u32 candidate_count = flattened;
    const u32 candidate_base = beam_count;
    for (u32 index = threadIdx.x; index < candidate_count; index += blockDim.x) {
      u32 lane = 0;
      u32 relative = index;
      while (lane < selected_count && relative >= neighbor_counts[lane]) {
        relative -= neighbor_counts[lane++];
      }
      const u32 offset = lane * kPersistentMaxGraphDegree + relative;
      merge_handles[candidate_base + index] = neighbor_handles[offset];
      merge_distances[candidate_base + index] = neighbor_distances[offset];
      merge_expanded[candidate_base + index] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (u32 lane = 0; lane < selected_count; ++lane) {
        total_remote_reads += remote_reads_by_lane[lane];
        total_cache_hits += cache_hits_by_lane[lane];
      }
    }
    __syncthreads();
    merge_approximate_into_beam(
      merge_handles + candidate_base, merge_distances + candidate_base,
      candidate_count, beam_handles, beam_ids, beam_distances,
      beam_expanded, beam_count, traversal_capacity,
      merge_handles, merge_ids, merge_distances, merge_expanded);
    if (threadIdx.x == 0) {
      beam_phase_cycles += clock64() - phase_started_cycles;
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
    rerank_count = min(beam_count, params.final_rerank_width);
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
                     beam_count, &total_exact_reads, &total_exact_cache_hits,
                     params.final_rerank_width, true, merge_handles, merge_ids,
                     merge_distances, merge_expanded);
  if (threadIdx.x == 0) {
    exact_phase_cycles += clock64() - phase_started_cycles;
  }
  __syncthreads();

  add_delta_candidates(params, descriptor, query, query_lut,
                       beam_handles, beam_ids, beam_distances, beam_expanded, beam_count,
                       params.final_rerank_width);
  if (beam_count == 0) {
    if (threadIdx.x == 0) {
      completion.status = -EIO;
      completion.gpu_cycles = clock64() - query_started_cycles;
      completion.remote_pages = total_remote_reads;
      completion.cache_hits = total_cache_hits;
      completion.exact_vectors = total_exact_reads;
      completion.exact_cache_hits = total_exact_cache_hits;
      device_ring_push(params.completions, completion);
    }
    __syncthreads();
    return;
  }
  if (threadIdx.x == 0) {
    for (u32 index = 0; index < beam_count; ++index) {
      u32 best = index;
      for (u32 candidate = index + 1; candidate < beam_count; ++candidate) {
        if (beam_distances[candidate] < beam_distances[best]) best = candidate;
      }
      if (best != index) {
        const u32 handle = beam_handles[index];
        beam_handles[index] = beam_handles[best];
        beam_handles[best] = handle;
        const u32 id = beam_ids[index];
        beam_ids[index] = beam_ids[best];
        beam_ids[best] = id;
        const f32 distance = beam_distances[index];
        beam_distances[index] = beam_distances[best];
        beam_distances[best] = distance;
      }
    }
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
    completion.cache_hits = total_cache_hits;
    completion.exact_vectors = total_exact_reads;
    completion.exact_cache_hits = total_exact_cache_hits;
    device_ring_push(params.completions, completion);
  }
}

__global__ void persistent_search_kernel(PersistentKernelParams params) {
  __shared__ QueryDescriptor descriptor;
  __shared__ u32 have_submission;
  __shared__ u32 stop_requested;
  for (;;) {
    if (threadIdx.x == 0) {
      stop_requested = *reinterpret_cast<volatile u32*>(params.stop);
    }
    __syncthreads();
    if (stop_requested != 0) return;
    if (threadIdx.x == 0) {
      have_submission = device_ring_try_pop(params.submissions, descriptor) ? 1u : 0u;
    }
    __syncthreads();
    if (have_submission == 0) {
      device_ring_relax(256);
      continue;
    }
    process_query(params, descriptor);
    __syncthreads();
  }
}

__global__ void publish_delta_count_kernel(u32* count, u32 value) {
  if (threadIdx.x == 0) atomicExch(count, value);
}

__global__ void supersede_delta_record_kernel(DeviceDeltaRecord* records,
                                              u32 slot, u64 epoch) {
  if (threadIdx.x == 0) {
    atomicExch(reinterpret_cast<unsigned long long*>(&records[slot].superseded_epoch), epoch);
  }
}

__global__ void insert_override_kernel(u32* keys, u64* epochs, u32 capacity,
                                       u32 ordinal, u64 epoch) {
  if (threadIdx.x != 0 || capacity == 0) return;
  const u32 mask = capacity - 1;
  u32 position = hash32(ordinal) & mask;
  for (u32 probe = 0; probe < capacity; ++probe) {
    const u32 old = atomicCAS(keys + position, UINT32_MAX, ordinal);
    if (old == UINT32_MAX || old == ordinal) {
      if (old == UINT32_MAX) atomicExch(
        reinterpret_cast<unsigned long long*>(epochs + position), epoch);
      else atomicMin(reinterpret_cast<unsigned long long*>(epochs + position), epoch);
      return;
    }
    position = (position + 1) & mask;
  }
}

__global__ void insert_remote_kernel(u64* keys, u32* slots, u32 capacity,
                                     u64 remote_node, u32 slot) {
  if (threadIdx.x != 0 || capacity == 0 || remote_node == 0) return;
  const u32 mask = capacity - 1;
  u32 position = hash64(remote_node) & mask;
  for (u32 probe = 0; probe < capacity; ++probe) {
    const u64 old = atomicCAS(reinterpret_cast<unsigned long long*>(keys + position),
                              0ULL, remote_node);
    if (old == 0 || old == remote_node) {
      atomicExch(slots + position, slot);
      return;
    }
    position = (position + 1) & mask;
  }
}

__global__ void link_bucket_kernel(u32* heads, u32* next, u32 bucket, u32 slot) {
  if (threadIdx.x == 0) next[slot] = atomicExch(heads + bucket, slot);
}

__global__ void invalidate_graph_cache_kernel(const u64* invalidation_keys,
                                              u32 invalidation_count,
                                              const u64* cache_keys,
                                              u32* cache_states,
                                              const u32* cache_readers,
                                              u32 cache_sets, u32 cache_ways) {
  for (u32 index = blockIdx.x * blockDim.x + threadIdx.x;
       index < invalidation_count; index += blockDim.x * gridDim.x) {
    const u64 key = invalidation_keys[index];
    const u32 set = hash64(key) % cache_sets;
    for (u32 way = 0; way < cache_ways; ++way) {
      const u32 slot = set * cache_ways + way;
      for (;;) {
        const u32 state = *reinterpret_cast<volatile u32*>(cache_states + slot);
        if (load_cg(cache_keys + slot) != key || state == 0) break;
        if (state == 1) {
          device_ring_relax(128);
          continue;
        }
        if (state != 2 || atomicCAS(cache_states + slot, 2u, 1u) != 2u) continue;
        while (*reinterpret_cast<const volatile u32*>(cache_readers + slot) != 0) {
          device_ring_relax(128);
        }
        __threadfence();
        atomicExch(cache_states + slot, 0u);
        break;
      }
    }
  }
}

__global__ void gpunetio_locked_read_probe_kernel(PersistentKernelParams params,
                                                   u8* destinations,
                                                   u32 destination_stride,
                                                   i32* statuses,
                                                   u32* completed,
                                                   u32 iterations) {
  constexpr u32 warp_width = 32;
  if (threadIdx.x % warp_width != 0) return;
  const u32 worker = threadIdx.x / warp_width;
  const u32 worker_count = min(params.direct_qps_per_node, blockDim.x / warp_width);
  if (worker >= worker_count) return;
  const u32 stream = blockIdx.x * worker_count + worker;
  i32 status = 0;
  for (u32 iteration = 0; iteration < iterations && status == 0; ++iteration) {
    status = direct_fetch(
      params, blockIdx.x % params.direct_region_count, 0,
      destinations + static_cast<size_t>(stream) * destination_stride,
      sizeof(u64), stream % params.direct_qps_per_node);
    if (status == 0) atomicAdd(completed, 1u);
  }
  statuses[stream] = status;
}

__global__ void gpunetio_batched_read_probe_kernel(PersistentKernelParams params,
                                                    u8* destinations,
                                                    u32 destination_stride,
                                                    i32* statuses,
                                                    u32* completed,
                                                    u32 batch_size) {
  __shared__ u32 request_shards[kPersistentMaxExact];
  __shared__ u64 remote_offsets[kPersistentMaxExact];
  const u32 memory_node = blockIdx.x % params.direct_region_count;
  for (u32 index = threadIdx.x; index < batch_size; index += blockDim.x) {
    request_shards[index] = memory_node;
    remote_offsets[index] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    i32 status = direct_fetch_batch(
      params, memory_node, request_shards, remote_offsets, batch_size,
      destinations + static_cast<size_t>(blockIdx.x) * batch_size * destination_stride,
      destination_stride, sizeof(u64), blockIdx.x % params.direct_qps_per_node);
    if (status == 0) {
      status = direct_fetch(
        params, memory_node, 0,
        destinations + static_cast<size_t>(blockIdx.x) * batch_size * destination_stride,
        sizeof(u64), blockIdx.x % params.direct_qps_per_node);
    }
    statuses[blockIdx.x] = status;
    if (status == 0) atomicAdd(completed, batch_size + 1);
  }
}

}  // namespace

void launch_persistent_search(cudaStream_t stream, const PersistentKernelParams& params,
                              u32 blocks, u32 threads) {
  persistent_search_kernel<<<blocks, threads, 0, stream>>>(params);
}

void launch_gpunetio_locked_read_probe(cudaStream_t stream,
                                       const PersistentKernelParams& params,
                                       u8* destinations, u32 destination_stride,
                                       i32* statuses, u32* completed,
                                       u32 blocks, u32 iterations) {
  gpunetio_locked_read_probe_kernel<<<blocks, 128, 0, stream>>>(
    params, destinations, destination_stride, statuses, completed, iterations);
}

void launch_gpunetio_batched_read_probe(cudaStream_t stream,
                                        const PersistentKernelParams& params,
                                        u8* destinations, u32 destination_stride,
                                        i32* statuses, u32* completed,
                                        u32 blocks, u32 batch_size) {
  gpunetio_batched_read_probe_kernel<<<blocks, 128, 0, stream>>>(
    params, destinations, destination_stride, statuses, completed, batch_size);
}

void launch_publish_delta_count(cudaStream_t stream, u32* count, u32 value) {
  publish_delta_count_kernel<<<1, 1, 0, stream>>>(count, value);
}

void launch_supersede_delta_record(cudaStream_t stream, DeviceDeltaRecord* records,
                                   u32 slot, u64 epoch) {
  supersede_delta_record_kernel<<<1, 1, 0, stream>>>(records, slot, epoch);
}

void launch_insert_base_override(cudaStream_t stream, u32* keys, u64* epochs,
                                 u32 capacity, u32 ordinal, u64 epoch) {
  insert_override_kernel<<<1, 1, 0, stream>>>(keys, epochs, capacity, ordinal, epoch);
}

void launch_insert_delta_remote(cudaStream_t stream, u64* keys, u32* slots,
                                u32 capacity, u64 remote_node, u32 slot) {
  insert_remote_kernel<<<1, 1, 0, stream>>>(keys, slots, capacity, remote_node, slot);
}

void launch_link_delta_bucket(cudaStream_t stream, u32* bucket_heads, u32* next,
                              u32 bucket, u32 slot) {
  link_bucket_kernel<<<1, 1, 0, stream>>>(bucket_heads, next, bucket, slot);
}

void launch_invalidate_graph_cache(cudaStream_t stream, const u64* invalidation_keys,
                                   u32 invalidation_count, const u64* cache_keys,
                                   u32* cache_states, const u32* cache_readers,
                                   u32 cache_sets, u32 cache_ways) {
  if (invalidation_count == 0 || cache_sets == 0 || cache_ways == 0) return;
  constexpr u32 threads = 128;
  const u32 blocks = std::min<u32>(256, (invalidation_count + threads - 1) / threads);
  invalidate_graph_cache_kernel<<<blocks, threads, 0, stream>>>(
    invalidation_keys, invalidation_count, cache_keys, cache_states,
    cache_readers, cache_sets, cache_ways);
}

}  // namespace gpu_search
