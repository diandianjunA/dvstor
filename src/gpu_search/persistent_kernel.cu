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

__device__ __forceinline__ i32 load_system_acquire(const i32* address) {
  u32 value = 0;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(address)
               : "memory");
  return static_cast<i32>(value);
}

__device__ __forceinline__ void store_system_release(i32* address, i32 value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;"
               :
               : "l"(address), "r"(static_cast<u32>(value))
               : "memory");
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
      doca_gpu_dev_verbs_load_relaxed<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
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
          u64, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
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
                                 const f32* query_lut, f32 query_norm2,
                                 const u8* entry) {
  f32 signed_dot = 0.0f;
  for (u32 byte = 0; byte < params.code_bits / 8; ++byte) {
    signed_dot += query_lut[static_cast<size_t>(byte) * 256 + entry[byte]];
  }
  f32 norm = 0.0f;
  f32 error = 1.0f;
  memcpy(&norm, entry + params.code_storage_bytes, sizeof(norm));
  memcpy(&error, entry + params.code_storage_bytes + sizeof(norm), sizeof(error));
  if (!isfinite(norm) || norm < 0.0f || !isfinite(error) || error <= 0.0f) return FLT_MAX;
  const f32 denominator = sqrtf(static_cast<f32>(params.code_bits)) * fmaxf(error, 1e-6f);
  const f32 inner_product = norm * signed_dot / denominator;
  return fmaxf(query_norm2 + norm * norm - 2.0f * inner_product, 0.0f);
}

__device__ f32 approximate_handle(const PersistentKernelParams& params,
                                  const f32* query_lut, f32 query_norm2,
                                  u32 handle, u64 snapshot_epoch) {
  if ((handle & kDeltaHandleBit) == 0) {
    if (handle >= params.num_nodes) return FLT_MAX;
    const u64 override_epoch = base_override_epoch(params, handle);
    if (override_epoch != 0 && override_epoch <= snapshot_epoch) return FLT_MAX;
    return approximate_entry(params, query_lut, query_norm2,
      params.rabitq_entries + static_cast<size_t>(handle) * params.rabitq_entry_bytes);
  }
  const u32 slot = handle & kDeltaHandleMask;
  if (slot >= min(load_cg(params.delta_count), params.delta_capacity)) return FLT_MAX;
  const DeviceDeltaRecord& record = params.delta_records[slot];
  if (!delta_visible(record, snapshot_epoch)) return FLT_MAX;
  return approximate_entry(params, query_lut, query_norm2,
    params.delta_rabitq_entries + static_cast<size_t>(slot) * params.rabitq_entry_bytes);
}

__device__ void beam_insert(u32* handles, u32* ids, f32* distances, u8* expanded,
                            u32& count, u32 capacity, u32 handle, u32 id, f32 distance) {
  if (handle == UINT32_MAX || id == UINT32_MAX ||
      !isfinite(distance) || distance == FLT_MAX) return;
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
      params.direct_qps_per_node == 0 || params.direct_disabled == nullptr ||
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
  doca_gpu_dev_verbs_ticket_t ticket = 0;
  doca_gpu_dev_verbs_get<DOCA_GPUNETIO_VERBS_DUMP_AUTO,
                         DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                         DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(
    qp, remote, local, bytes, dump, &ticket);
  auto* completion_queue = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
  const i32 status = poll_direct_cq(completion_queue, ticket, params.direct_timeout_ns,
                                    params.stop, params.direct_disabled);
  if (status != 0) {
    if (params.direct_error != nullptr) atomicCAS(params.direct_error, 0, status);
    atomicExch(params.direct_disabled, 1u);
  }
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

__device__ bool fetch_remote_exact_record(const PersistentKernelParams& params,
                                          const QueryDescriptor& descriptor,
                                          u32 shard, u64 node_offset,
                                          u8* destination, u32 staging_index,
                                          u32 lane, u32* exact_reads) {
  const u32 status_index = descriptor.query_slot * params.fetch_status_stride + staging_index;
  store_system_release(params.fetch_status + status_index, 0);
  if (params.direct_backend != 0) {
    if (*reinterpret_cast<volatile u32*>(params.direct_disabled) != 0) return false;
    const i32 status = direct_fetch(params, shard, node_offset, destination,
                                    params.node_record_bytes, lane);
    if (status != 0) return false;
    store_system_release(params.fetch_status + status_index, 1);
  } else {
    store_system_release(params.fetch_status + status_index, 0);
    FetchDescriptor fetch{
      .request_id = descriptor.request_id,
      .remote_offset = node_offset,
      .destination_address = reinterpret_cast<u64>(destination),
      .bytes = params.node_record_bytes,
      .memory_node = static_cast<u16>(shard),
      .kind = static_cast<u8>(FetchKind::node_record),
      .sequence = status_index,
    };
    device_ring_push(params.fetches, fetch);
    while (load_system_acquire(params.fetch_status + status_index) == 0 &&
           *reinterpret_cast<volatile u32*>(params.stop) == 0) {
      device_ring_relax(128);
    }
  }
  atomicAdd(exact_reads, 1u);
  return load_system_acquire(params.fetch_status + status_index) > 0;
}

__device__ const u8* acquire_exact_record(const PersistentKernelParams& params,
                                          const QueryDescriptor& descriptor,
                                          u32 handle, u32 shard, u64 node_offset,
                                          u32 staging_index, u32 lane,
                                          u32* exact_reads, u32* cache_hits,
                                          u32& acquired_slot) {
  acquired_slot = UINT32_MAX;
  if (params.exact_cache_sets == 0 || params.exact_cache_ways == 0) {
    u8* destination = params.exact_records +
      (static_cast<size_t>(descriptor.query_slot) * params.exact_width + staging_index) *
        params.node_record_bytes;
    return fetch_remote_exact_record(params, descriptor, shard, node_offset,
                                     destination, staging_index, lane, exact_reads)
      ? destination : nullptr;
  }

  const u32 set = hash32(handle) % params.exact_cache_sets;
  for (;;) {
    bool retry_lookup = false;
    for (u32 way = 0; way < params.exact_cache_ways; ++way) {
      const u32 slot = set * params.exact_cache_ways + way;
      const u32 state = *reinterpret_cast<volatile u32*>(params.exact_cache_states + slot);
      if (state == 2 && load_cg(params.exact_cache_keys + slot) == handle) {
        atomicAdd(params.exact_cache_readers + slot, 1u);
        __threadfence();
        if (*reinterpret_cast<volatile u32*>(params.exact_cache_states + slot) == 2 &&
            load_cg(params.exact_cache_keys + slot) == handle) {
          atomicAdd(cache_hits, 1u);
          acquired_slot = slot;
          return params.exact_cache + static_cast<size_t>(slot) * params.exact_cache_stride;
        }
        atomicSub(params.exact_cache_readers + slot, 1u);
      }
      if (state == 1 && load_cg(params.exact_cache_keys + slot) == handle) {
        while (*reinterpret_cast<volatile u32*>(params.exact_cache_states + slot) == 1 &&
               *reinterpret_cast<volatile u32*>(params.stop) == 0) {
          device_ring_relax(128);
        }
        retry_lookup = true;
        break;
      }
    }
    if (retry_lookup) continue;

    const u32 start_way = atomicAdd(params.exact_cache_victims + set, 1u) %
      params.exact_cache_ways;
    for (u32 attempt = 0; attempt < params.exact_cache_ways; ++attempt) {
      const u32 slot = set * params.exact_cache_ways +
        (start_way + attempt) % params.exact_cache_ways;
      const u32 state = *reinterpret_cast<volatile u32*>(params.exact_cache_states + slot);
      if (state == 1 || atomicCAS(params.exact_cache_states + slot, state, 1u) != state) continue;
      params.exact_cache_keys[slot] = handle;
      __threadfence();
      while (*reinterpret_cast<volatile u32*>(params.exact_cache_readers + slot) != 0 &&
             *reinterpret_cast<volatile u32*>(params.stop) == 0) {
        device_ring_relax(128);
      }
      u8* destination = params.exact_cache +
        static_cast<size_t>(slot) * params.exact_cache_stride;
      if (!fetch_remote_exact_record(params, descriptor, shard, node_offset,
                                     destination, staging_index, lane, exact_reads)) {
        __threadfence();
        atomicExch(params.exact_cache_states + slot, 0u);
        return nullptr;
      }
      params.exact_cache_readers[slot] = 1;
      __threadfence();
      atomicExch(params.exact_cache_states + slot, 2u);
      acquired_slot = slot;
      return destination;
    }
    if (*reinterpret_cast<volatile u32*>(params.stop) != 0) return nullptr;
    device_ring_relax(128);
  }
}

__device__ f32 fetch_exact_candidate(const PersistentKernelParams& params,
                                     const QueryDescriptor& descriptor,
                                     const f32* query, u32 handle,
                                     u32 staging_index, u32 lane,
                                     u32* id_out, u32* exact_reads,
                                     u32* exact_cache_hits) {
  if (id_out != nullptr) *id_out = UINT32_MAX;
  if ((handle & kDeltaHandleBit) != 0) {
    const u32 delta_slot = handle & kDeltaHandleMask;
    if (delta_slot >= min(load_cg(params.delta_count), params.delta_capacity)) return FLT_MAX;
    const DeviceDeltaRecord& record = params.delta_records[delta_slot];
    if (!delta_visible(record, descriptor.snapshot_epoch)) return FLT_MAX;
    if (id_out != nullptr) *id_out = record.id;
    return exact_float_distance(
      params, query, params.delta_vectors + static_cast<size_t>(delta_slot) * params.dim);
  }
  const u64 override_epoch = base_override_epoch(params, handle);
  if (override_epoch != 0 && override_epoch <= descriptor.snapshot_epoch) return FLT_MAX;
  if (staging_index >= params.exact_width) return FLT_MAX;
  u64 raw = 0;
  u64 graph_offset = 0;
  u32 shard = 0;
  if (!resolve_handle(params, handle, raw, shard, graph_offset)) return FLT_MAX;
  const u64 node_offset = ((raw << 16) >> 16) + params.node_meta_offset;
  u32 acquired_slot = UINT32_MAX;
  const u8* record = acquire_exact_record(
    params, descriptor, handle, shard, node_offset, staging_index, lane,
    exact_reads, exact_cache_hits, acquired_slot);
  if (record == nullptr) return FLT_MAX;
  if (id_out != nullptr) *id_out = *reinterpret_cast<const u32*>(record);
  const f32 distance = exact_storage_distance(params, query, record + 8);
  if (acquired_slot != UINT32_MAX) {
    __threadfence();
    atomicSub(params.exact_cache_readers + acquired_slot, 1u);
  }
  return distance;
}

__device__ void gate_insert(u32* handles, f32* distances, u32& count,
                            u32 capacity, u32 handle, f32 distance) {
  if (handle == UINT32_MAX || !isfinite(distance) || distance == FLT_MAX || capacity == 0) return;
  if (count < capacity) {
    handles[count] = handle;
    distances[count] = distance;
    ++count;
    return;
  }
  u32 worst = 0;
  for (u32 index = 1; index < count; ++index) {
    if (distances[index] > distances[worst]) worst = index;
  }
  if (distance >= distances[worst]) return;
  handles[worst] = handle;
  distances[worst] = distance;
}

__device__ void exactify_into_beam(const PersistentKernelParams& params,
                                   const QueryDescriptor& descriptor,
                                   const f32* query, u32* candidate_handles,
                                   u32* candidate_ids, f32* candidate_distances,
                                   u32 candidate_count, u32* beam_handles,
                                   u32* beam_ids, f32* beam_distances,
                                   u8* beam_expanded, u32& beam_count,
                                   u32* exact_reads, u32* exact_cache_hits) {
  for (u32 index = threadIdx.x; index < candidate_count; index += blockDim.x) {
    candidate_distances[index] = fetch_exact_candidate(
      params, descriptor, query, candidate_handles[index], index,
      descriptor.query_slot * params.exact_width + index, candidate_ids + index,
      exact_reads, exact_cache_hits);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (u32 index = 0; index < candidate_count; ++index) {
      beam_insert(beam_handles, beam_ids, beam_distances, beam_expanded, beam_count,
                  params.beam_width, candidate_handles[index], candidate_ids[index],
                  candidate_distances[index]);
    }
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
                                        u32 status_index, u32& remote_reads,
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
        store_system_release(params.fetch_status + status_index, 0);
        const bool direct_backend = params.direct_backend != 0;
        bool fetched_direct = false;
        if (direct_backend) {
          if (*reinterpret_cast<volatile u32*>(params.direct_disabled) != 0) break;
          const i32 status = direct_fetch(params, shard, graph_offset, destination,
                                          params.graph_entry_bytes, request_lane + retry);
          fetched_direct = status == 0;
          if (!fetched_direct) break;
          store_system_release(params.fetch_status + status_index, 1);
        }
        if (!direct_backend) {
          store_system_release(params.fetch_status + status_index, 0);
          FetchDescriptor fetch{
            .request_id = descriptor.request_id,
            .remote_offset = graph_offset,
            .destination_address = reinterpret_cast<u64>(destination),
            .bytes = params.graph_entry_bytes,
            .memory_node = static_cast<u16>(shard),
            .kind = static_cast<u8>(FetchKind::graph_record),
            .sequence = status_index,
          };
          device_ring_push(params.fetches, fetch);
          while (load_system_acquire(params.fetch_status + status_index) == 0 &&
                 *reinterpret_cast<volatile u32*>(params.stop) == 0) {
            device_ring_relax(128);
          }
        }
        ++remote_reads;
        ready = load_system_acquire(params.fetch_status + status_index) > 0 &&
          valid_graph_record(params, destination);
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
                                     f32 query_norm2, u32* beam_handles,
                                     u32* beam_ids, f32* beam_distances,
                                     u8* beam_expanded, u32& beam_count) {
  __shared__ u32 delta_count_snapshot;
  if (threadIdx.x == 0) {
    delta_count_snapshot = min(load_cg(params.delta_count), params.delta_capacity);
  }
  __syncthreads();
  const u32 count = delta_count_snapshot;
  if (count == 0) return;
  __shared__ u32 candidate_handles[128];
  __shared__ f32 candidate_distances[128];
  u32 local_slot = UINT32_MAX;
  f32 local_approximation = FLT_MAX;

  if (params.anchor_count == 0 || count <= 4096) {
    for (u32 slot = threadIdx.x; slot < count; slot += blockDim.x) {
      const DeviceDeltaRecord& record = params.delta_records[slot];
      if (!delta_visible(record, descriptor.snapshot_epoch)) continue;
      const f32 approximation = approximate_entry(
        params, query_lut, query_norm2,
        params.delta_rabitq_entries + static_cast<size_t>(slot) * params.rabitq_entry_bytes);
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
    __shared__ u32 anchor_best_indices[128];
    __shared__ f32 anchor_best_distances[128];
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
            params, query_lut, query_norm2,
            params.delta_rabitq_entries + static_cast<size_t>(slot) * params.rabitq_entry_bytes);
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
    for (u32 index = 0; index < min(blockDim.x, 128u); ++index) {
      const u32 handle = candidate_handles[index];
      if (handle == UINT32_MAX) continue;
      bool duplicate = false;
      for (u32 beam = 0; beam < beam_count; ++beam) {
        if (beam_handles[beam] == handle) duplicate = true;
      }
      if (!duplicate) {
        const u32 slot = handle & kDeltaHandleMask;
        beam_insert(beam_handles, beam_ids, beam_distances, beam_expanded, beam_count,
                    params.beam_width, handle, params.delta_records[slot].id,
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
  f32* rotated = params.rotated_queries +
    static_cast<size_t>(query_slot) * params.code_bits;
  __shared__ f32 query_norm2;
  if (threadIdx.x == 0) query_norm2 = 0.0f;
  __syncthreads();
  f32 local_norm = 0.0f;
  for (u32 dimension = threadIdx.x; dimension < params.code_bits; dimension += blockDim.x) {
    f32 value = dimension < params.dim ? query[dimension] - params.centroid[dimension] : 0.0f;
    if (dimension < params.dim) {
      local_norm += value * value;
      u32 hash = dimension + 0x9e3779b9U;
      hash ^= hash >> 16;
      hash *= 0x7feb352dU;
      hash ^= hash >> 15;
      value = (hash & 1U) ? value : -value;
    }
    rotated[dimension] = value;
  }
  atomicAdd(&query_norm2, local_norm);
  __syncthreads();
  for (u32 width = 1; width < params.code_bits; width <<= 1) {
    for (u32 index = threadIdx.x; index < params.code_bits / 2; index += blockDim.x) {
      const u32 group = index / width;
      const u32 offset = index % width;
      const u32 lhs_index = group * (width << 1) + offset;
      const u32 rhs_index = lhs_index + width;
      const f32 lhs = rotated[lhs_index];
      const f32 rhs = rotated[rhs_index];
      rotated[lhs_index] = lhs + rhs;
      rotated[rhs_index] = lhs - rhs;
    }
    __syncthreads();
  }
  const f32 rotation_scale = rsqrtf(static_cast<f32>(params.code_bits));
  for (u32 dimension = threadIdx.x; dimension < params.code_bits; dimension += blockDim.x) {
    rotated[dimension] *= rotation_scale;
  }
  const u32 code_bytes = params.code_bits / 8;
  f32* query_lut = params.query_luts +
    static_cast<size_t>(query_slot) * code_bytes * 256;
  for (u32 index = threadIdx.x; index < code_bytes * 256; index += blockDim.x) {
    const u32 byte = index / 256;
    const u32 code = index & 255u;
    f32 signed_dot = 0.0f;
    for (u32 bit = 0; bit < 8; ++bit) {
      const f32 value = rotated[byte * 8 + bit];
      signed_dot += (code & (1u << (7u - bit))) != 0 ? value : -value;
    }
    query_lut[index] = signed_dot;
  }
  __syncthreads();

  u32* beam_handles = params.beam_handles +
    static_cast<size_t>(query_slot) * params.beam_width;
  u32* beam_ids = params.beam_ids +
    static_cast<size_t>(query_slot) * params.beam_width;
  f32* beam_distances = params.beam_distances +
    static_cast<size_t>(query_slot) * params.beam_width;
  u8* beam_expanded = params.beam_expanded +
    static_cast<size_t>(query_slot) * params.beam_width;
  u32* visited = params.visited_hash +
    static_cast<size_t>(query_slot) * params.visited_capacity;
  for (u32 index = threadIdx.x; index < params.beam_width; index += blockDim.x) {
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
  __shared__ f32 entry_distances[kPersistentMaxEntryPoints];
  __shared__ u32 gate_handles[kPersistentMaxExact];
  __shared__ u32 gate_ids[kPersistentMaxExact];
  __shared__ f32 gate_distances[kPersistentMaxExact];
  __shared__ u32 gate_count;
  __shared__ u32 total_exact_reads;
  __shared__ u32 total_exact_cache_hits;
  for (u32 index = threadIdx.x; index < params.entry_point_count; index += blockDim.x) {
    entry_distances[index] = approximate_handle(
      params, query_lut, query_norm2, params.entry_points[index], descriptor.snapshot_epoch);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    beam_count = 0;
    gate_count = 0;
    total_exact_reads = 0;
    total_exact_cache_hits = 0;
    const u32 seed_capacity = min(params.exact_width, params.beam_width);
    for (u32 index = 0; index < params.entry_point_count; ++index) {
      gate_insert(gate_handles, gate_distances, gate_count, seed_capacity,
                  params.entry_points[index], entry_distances[index]);
    }
    if (gate_count == 0) gate_handles[gate_count++] = params.medoid_ordinal;
  }
  __syncthreads();
  exactify_into_beam(params, descriptor, query, gate_handles, gate_ids, gate_distances,
                     gate_count, beam_handles, beam_ids, beam_distances, beam_expanded, beam_count,
                     &total_exact_reads, &total_exact_cache_hits);
  if (threadIdx.x == 0) {
    for (u32 index = 0; index < beam_count; ++index) {
      insert_visited(visited, params.visited_capacity, beam_handles[index]);
    }
  }
  __syncthreads();
  if (beam_count == 0) {
    if (threadIdx.x == 0) {
      completion.status = -EIO;
      completion.gpu_cycles = clock64() - query_started_cycles;
      completion.exact_vectors = total_exact_reads;
      completion.exact_cache_hits = total_exact_cache_hits;
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
  __shared__ u32 lane_force_exact[kPersistentMaxPrefetch];
  if (threadIdx.x == 0) expansions = 0;
  __syncthreads();
  while (expansions < params.max_expansions) {
    if (threadIdx.x == 0) {
      selected_count = 0;
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
    if (threadIdx.x < selected_count) {
      const u32 lane = threadIdx.x;
      remote_reads_by_lane[lane] = 0;
      cache_hits_by_lane[lane] = 0;
      const u32 status_index = query_slot * params.fetch_status_stride +
        params.exact_width + lane;
      u32 acquired_slot = UINT32_MAX;
      const u8* record = fetch_graph_record(
        params, descriptor, selected_handles[lane],
        query_slot * params.prefetch_depth + lane, status_index,
        remote_reads_by_lane[lane], cache_hits_by_lane[lane], acquired_slot);
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
        ? approximate_handle(params, query_lut, query_norm2,
                             handle, descriptor.snapshot_epoch)
        : FLT_MAX;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (u32 lane = 0; lane < selected_count; ++lane) {
        total_remote_reads += remote_reads_by_lane[lane];
        total_cache_hits += cache_hits_by_lane[lane];
        const u32 expansion_number = expansions + lane + 1;
        lane_force_exact[lane] = expansion_number <= params.warmup_exact_expansions ||
          (params.audit_period != 0 &&
           expansion_number > params.warmup_exact_expansions &&
           (expansion_number - params.warmup_exact_expansions) % params.audit_period == 0);
      }
    }
    __syncthreads();

    for (u32 lane = 0; lane < selected_count; ++lane) {
      if (lane_force_exact[lane] == 0) continue;
      if (threadIdx.x == 0) gate_count = min(params.exact_width, neighbor_counts[lane]);
      __syncthreads();
      for (u32 index = threadIdx.x; index < gate_count; index += blockDim.x) {
        const u32 offset = lane * kPersistentMaxGraphDegree + index;
        gate_handles[index] = isfinite(neighbor_distances[offset])
          ? neighbor_handles[offset] : UINT32_MAX;
      }
      __syncthreads();
      exactify_into_beam(params, descriptor, query, gate_handles, gate_ids, gate_distances,
                         gate_count, beam_handles, beam_ids, beam_distances, beam_expanded,
                         beam_count, &total_exact_reads, &total_exact_cache_hits);
    }

    if (threadIdx.x == 0) {
      gate_count = 0;
      for (u32 lane = 0; lane < selected_count; ++lane) {
        if (lane_force_exact[lane] != 0) continue;
        const u32 segment_begin = gate_count;
        u32 segment_count = 0;
        const u32 segment_base = min(neighbor_counts[lane], params.gate_width);
        const u32 segment_limit = min(
          params.exact_width - gate_count,
          min(neighbor_counts[lane], max(segment_base, params.gate_max_width)));
        for (u32 neighbor = 0; neighbor < neighbor_counts[lane]; ++neighbor) {
          const u32 offset = lane * kPersistentMaxGraphDegree + neighbor;
          gate_insert(gate_handles + segment_begin, gate_distances + segment_begin,
                      segment_count, segment_limit,
                      neighbor_handles[offset], neighbor_distances[offset]);
        }
        for (u32 index = 0; index < segment_count; ++index) {
          u32 best = index;
          for (u32 candidate = index + 1; candidate < segment_count; ++candidate) {
            if (gate_distances[segment_begin + candidate] <
                gate_distances[segment_begin + best]) best = candidate;
          }
          if (best != index) {
            const u32 handle = gate_handles[segment_begin + index];
            gate_handles[segment_begin + index] = gate_handles[segment_begin + best];
            gate_handles[segment_begin + best] = handle;
            const f32 distance = gate_distances[segment_begin + index];
            gate_distances[segment_begin + index] = gate_distances[segment_begin + best];
            gate_distances[segment_begin + best] = distance;
          }
        }
        const u32 base_count = min(segment_count, segment_base);
        u32 selected_gate_count = base_count;
        if (base_count != 0) {
          const f32 cutoff = gate_distances[segment_begin + base_count - 1];
          const f32 margin_cutoff = cutoff + fabsf(cutoff) * fmaxf(params.gate_margin, 0.0f);
          while (selected_gate_count < segment_count &&
                 gate_distances[segment_begin + selected_gate_count] <= margin_cutoff) {
            ++selected_gate_count;
          }
        }
        gate_count += selected_gate_count;
      }
    }
    __syncthreads();
    if (gate_count != 0) {
      exactify_into_beam(params, descriptor, query, gate_handles, gate_ids, gate_distances,
                         gate_count, beam_handles, beam_ids, beam_distances, beam_expanded,
                         beam_count, &total_exact_reads, &total_exact_cache_hits);
    }
    if (threadIdx.x == 0) expansions += selected_count;
    __syncthreads();
  }

  add_delta_candidates(params, descriptor, query, query_lut, query_norm2,
                       beam_handles, beam_ids, beam_distances, beam_expanded, beam_count);
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

}  // namespace

void launch_persistent_search(cudaStream_t stream, const PersistentKernelParams& params,
                              u32 blocks, u32 threads) {
  persistent_search_kernel<<<blocks, threads, 0, stream>>>(params);
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
