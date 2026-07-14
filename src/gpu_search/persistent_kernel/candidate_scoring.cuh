#pragma once

#include "gpu_search/persistent_kernel/context.cuh"

namespace gpu_search::persistent_kernel_detail {

struct CandidateWorkspaceArrays {
  u32 handles[kPersistentMaxExact * 2];
  f32 distances[kPersistentMaxExact * 2];
  u32 ids[kPersistentMaxExact * 2];
  u8 expanded[kPersistentMaxExact * 2];
};

union CandidateSortWorkspace {
  ApproximateBlockSortWide::TempStorage radix_sort_wide;
  ApproximateBlockSortCompactPass::TempStorage radix_sort_compact_pass;
  ApproximateBlockSortCompactFinal::TempStorage radix_sort_compact_final;
};

struct CandidateWorkspace {
  CandidateWorkspaceArrays arrays;
  CandidateSortWorkspace sort;
};

__device__ void unlink_mutable_delta(const PersistentKernelParams& params,
                                     u32 slot) {
  if (slot >= params.delta_capacity || params.delta_prev == nullptr ||
      params.delta_next == nullptr || params.delta_bucket_heads == nullptr) {
    return;
  }
  const DeviceDeltaRecord record = params.delta_records[slot];
  if (record.anchor_bucket >= params.anchor_count) return;
  const u32 previous = params.delta_prev[slot];
  const u32 next = params.delta_next[slot];
  if (previous == UINT32_MAX) {
    atomicCAS(params.delta_bucket_heads + record.anchor_bucket, slot, next);
  } else if (previous < params.delta_capacity) {
    atomicCAS(params.delta_next + previous, slot, next);
  }
  if (next < params.delta_capacity) {
    atomicCAS(params.delta_prev + next, slot, previous);
  }
  params.delta_prev[slot] = UINT32_MAX;
  params.delta_next[slot] = UINT32_MAX;
}

constexpr u32 kGraphCacheEmpty = 0;
constexpr u32 kGraphCacheFilling = 1;
constexpr u32 kGraphCacheReady = 2;
constexpr u32 kGraphCacheStale = 3;
constexpr u32 kGraphCacheFillInvalidated = 4;
constexpr u32 kGraphScratchBit = 0x80000000u;
constexpr u32 kGraphRouteBit = 0x40000000u;
constexpr u32 kGraphSlotMask = ~(kGraphScratchBit | kGraphRouteBit);
constexpr u32 kCacheAdmissionWays = 4;
constexpr u32 kCacheWaitRounds = 64;
constexpr u64 kNodeLockMask = 1ull;
constexpr u64 kNodeDeletedMask = 1ull << 24;
constexpr u32 kNodeIdOffset = 8;
constexpr u32 kNodeVectorOffset = 16;

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

__device__ u32 anchor_graph_slot(const PersistentKernelParams& params, u64 key) {
  if (params.anchor_graph_keys == nullptr || params.anchor_graph_count == 0) {
    return UINT32_MAX;
  }
  u32 begin = 0;
  u32 end = params.anchor_graph_count;
  while (begin < end) {
    const u32 middle = begin + (end - begin) / 2;
    const u64 candidate = load_cg(params.anchor_graph_keys + middle);
    if (candidate < key) begin = middle + 1;
    else end = middle;
  }
  return begin < params.anchor_graph_count &&
      load_cg(params.anchor_graph_keys + begin) == key
    ? begin : UINT32_MAX;
}

__device__ void release_graph_record(const PersistentKernelParams& params,
                                     u32 acquired_slot) {
  if (acquired_slot == UINT32_MAX ||
      (acquired_slot & kGraphScratchBit) != 0) {
    return;
  }
  if ((acquired_slot & kGraphRouteBit) != 0) {
    const u32 slot = acquired_slot & kGraphSlotMask;
    atomicSub(params.anchor_graph_readers + slot, 1u);
    return;
  }
  atomicSub(params.graph_cache_readers + acquired_slot, 1u);
}

__device__ __forceinline__ u64 global_time_ns() {
  u64 value = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ f32 query_component(const u8* query, u8 dtype,
                                                u32 index) {
  switch (dtype) {
    case 0:
      return reinterpret_cast<const f32*>(query)[index];
    case 1:
      return static_cast<f32>(query[index]);
    case 2:
      return static_cast<f32>(reinterpret_cast<const std::int8_t*>(query)[index]);
    default:
      return 0.0f;
  }
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
    if (key == kDeltaRemoteEmpty) return UINT32_MAX;
    position = (position + 1) & mask;
  }
  return UINT32_MAX;
}

__device__ u32 resident_pq_slot_from_raw(const PersistentKernelParams& params,
                                         u64 raw) {
  if (raw == 0 || params.resident_pq_table_capacity == 0 ||
      params.resident_pq_keys == nullptr || params.resident_pq_slots == nullptr) {
    return UINT32_MAX;
  }
  const u32 mask = params.resident_pq_table_capacity - 1;
  u32 position = hash64(raw) & mask;
  for (u32 probe = 0; probe < params.resident_pq_table_capacity; ++probe) {
    const u64 key = load_cg(params.resident_pq_keys + position);
    if (key == raw) {
      const u32 slot = load_cg(params.resident_pq_slots + position);
      return slot < params.resident_pq_capacity ? slot : UINT32_MAX;
    }
    if (key == kDeltaRemoteEmpty) return UINT32_MAX;
    position = (position + 1) & mask;
  }
  return UINT32_MAX;
}

__device__ bool insert_resident_pq(const PersistentKernelParams& params,
                                   u64 raw, u32 slot) {
  if (raw == 0 || slot >= params.resident_pq_capacity ||
      params.resident_pq_table_capacity == 0 ||
      params.resident_pq_keys == nullptr || params.resident_pq_slots == nullptr ||
      params.resident_pq_positions == nullptr) {
    return false;
  }
  const u32 mask = params.resident_pq_table_capacity - 1;
  u32 position = hash64(raw) & mask;
  u32 first_tombstone = UINT32_MAX;
  for (u32 probe = 0; probe < params.resident_pq_table_capacity; ++probe) {
    const u64 key = load_cg(params.resident_pq_keys + position);
    if (key == raw) {
      atomicExch(params.resident_pq_slots + position, slot);
      params.resident_pq_positions[slot] = position;
      __threadfence();
      return true;
    }
    if (key == kDeltaRemoteTombstone && first_tombstone == UINT32_MAX) {
      first_tombstone = position;
    }
    if (key == kDeltaRemoteEmpty) {
      const u32 destination = first_tombstone == UINT32_MAX
        ? position : first_tombstone;
      params.resident_pq_positions[slot] = destination;
      params.resident_pq_slots[destination] = slot;
      __threadfence();
      atomicExch(reinterpret_cast<unsigned long long*>(
                   params.resident_pq_keys + destination), raw);
      return true;
    }
    position = (position + 1) & mask;
  }
  if (first_tombstone == UINT32_MAX) return false;
  params.resident_pq_positions[slot] = first_tombstone;
  params.resident_pq_slots[first_tombstone] = slot;
  __threadfence();
  atomicExch(reinterpret_cast<unsigned long long*>(
               params.resident_pq_keys + first_tombstone), raw);
  return true;
}

__device__ void erase_resident_pq(const PersistentKernelParams& params,
                                  const ResidentPqEraseUpdate& update) {
  if (update.remote_node == 0 || update.slot >= params.resident_pq_capacity ||
      params.resident_pq_positions == nullptr) {
    return;
  }
  const u32 position = params.resident_pq_positions[update.slot];
  if (position >= params.resident_pq_table_capacity ||
      load_cg(params.resident_pq_keys + position) != update.remote_node ||
      load_cg(params.resident_pq_slots + position) != update.slot) {
    return;
  }
  atomicCAS(reinterpret_cast<unsigned long long*>(params.resident_pq_keys + position),
            update.remote_node, kDeltaRemoteTombstone);
  atomicExch(params.resident_pq_slots + position, UINT32_MAX);
  atomicExch(params.resident_pq_positions + update.slot, UINT32_MAX);
}

__device__ u32 handle_from_raw(const PersistentKernelParams& params, u64 raw) {
  u32 handle = UINT32_MAX;
  if (static_handle_from_raw(params, raw, handle)) return handle;
  const u32 shard = static_cast<u32>(raw >> 48);
  const u64 offset = (raw << 16) >> 16;
  if (shard >= params.num_shards || params.graph_shard_bits >= 31) return UINT32_MAX;
  const DeviceShardRegion& region = params.shards[shard];
  if (offset < region.dynamic_base_offset || region.dynamic_record_bytes == 0) {
    return UINT32_MAX;
  }
  const u64 relative = offset - region.dynamic_base_offset;
  if (relative % region.dynamic_record_bytes != 0) return UINT32_MAX;
  const u64 slot = relative / region.dynamic_record_bytes;
  const u32 slot_bits = 31 - params.graph_shard_bits;
  const u64 slot_limit = 1ull << slot_bits;
  if (slot >= slot_limit || shard >= (1u << params.graph_shard_bits)) return UINT32_MAX;
  return kDeltaHandleBit |
    (shard << slot_bits) | static_cast<u32>(slot);
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
  if (params.graph_shard_bits >= 31) return false;
  const u32 slot_bits = 31 - params.graph_shard_bits;
  const u32 slot_mask = slot_bits == 31 ? kDeltaHandleMask : (1u << slot_bits) - 1u;
  shard = params.graph_shard_bits == 0
    ? 0u : (handle & kDeltaHandleMask) >> slot_bits;
  if (shard >= params.num_shards) return false;
  const DeviceShardRegion& region = params.shards[shard];
  if (region.dynamic_record_bytes == 0) return false;
  const u64 dynamic_slot = handle & slot_mask;
  const u64 node_offset = region.dynamic_base_offset +
    dynamic_slot * region.dynamic_record_bytes;
  if (node_offset < region.dynamic_base_offset || node_offset >= (1ull << 48)) return false;
  raw = (static_cast<u64>(shard) << 48) | node_offset;
  graph_offset = node_offset + region.dynamic_hot_offset;
  return true;
}

__device__ bool base_overridden(const PersistentKernelParams& params,
                                u32 ordinal, u64 snapshot_epoch) {
  const u32 word = ordinal / 32;
  if (params.permanent_override_bits != nullptr &&
      word < params.permanent_override_words &&
      (load_cg(params.permanent_override_bits + word) &
       (1u << (ordinal % 32))) != 0) {
    return true;
  }
  if (params.base_override_capacity == 0) return false;
  const u32 mask = params.base_override_capacity - 1;
  u32 position = hash32(ordinal) & mask;
  for (u32 probe = 0; probe < params.base_override_capacity; ++probe) {
    const u32 key = load_cg(params.base_override_keys + position);
    if (key == ordinal) {
      const u64 epoch = load_cg(params.base_override_epochs + position);
      return epoch != 0 && epoch <= snapshot_epoch;
    }
    if (key == kBaseOverrideEmpty) return false;
    position = (position + 1) & mask;
  }
  return false;
}

__device__ bool delta_visible(const DeviceDeltaRecord& record, u64 snapshot_epoch) {
  const u64 superseded = load_cg(&record.superseded_epoch);
  return record.epoch <= snapshot_epoch &&
    (superseded == 0 || superseded > snapshot_epoch) &&
    (record.flags & (kDeltaDeleted | kDeltaDurable)) == 0;
}

__device__ bool delta_code_visible(const DeviceDeltaRecord& record,
                                   u64 snapshot_epoch) {
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
    return approximate_entry(params, query_lut,
      params.pq_codes + static_cast<size_t>(handle) * params.pq_code_bytes);
  }
  u64 raw = 0;
  u64 graph_offset = 0;
  u32 shard = 0;
  if (!resolve_handle(params, handle, raw, shard, graph_offset)) return FLT_MAX;
  const u32 slot = delta_slot_from_raw(params, raw);
  if (slot < min(load_cg(params.delta_count), params.delta_capacity) &&
      params.delta_records[slot].remote_node == raw) {
    const DeviceDeltaRecord& record = params.delta_records[slot];
    if (delta_code_visible(record, snapshot_epoch)) {
      return approximate_entry(params, query_lut,
        params.delta_pq_codes + static_cast<size_t>(slot) * params.pq_code_bytes);
    }
    const u64 superseded = load_cg(&record.superseded_epoch);
    if (record.epoch <= snapshot_epoch &&
        ((record.flags & kDeltaDeleted) != 0 ||
         (superseded != 0 && superseded <= snapshot_epoch))) {
      return FLT_MAX;
    }
  }
  const u32 resident_slot = resident_pq_slot_from_raw(params, raw);
  if (resident_slot == UINT32_MAX || params.resident_pq_codes == nullptr) {
    return FLT_MAX;
  }
  return approximate_entry(params, query_lut,
    params.resident_pq_codes +
      static_cast<size_t>(resident_slot) * params.pq_code_bytes);
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

template <class BlockSort, u32 ItemsPerThread>
__device__ void merge_approximate_radix(
    u32* candidate_handles, f32* candidate_distances, u32 candidate_count,
    u32* beam_handles, u32* beam_ids, f32* beam_distances,
    u8* beam_expanded, u32& beam_count, u32 beam_capacity,
    u32 existing_count, u32 merge_count,
    typename BlockSort::TempStorage& radix_storage) {
  f32 local_distances[ItemsPerThread];
  u64 local_values[ItemsPerThread];
  for (u32 item = 0; item < ItemsPerThread; ++item) {
    const u32 index = threadIdx.x * ItemsPerThread + item;
    u32 handle = UINT32_MAX;
    u32 expanded = 0;
    f32 distance = FLT_MAX;
    if (index < existing_count) {
      handle = beam_handles[index];
      expanded = beam_expanded[index];
      distance = beam_distances[index];
    } else if (index < merge_count) {
      const u32 candidate = index - existing_count;
      handle = candidate_handles[candidate];
      distance = candidate_distances[candidate];
    }
    if (handle == UINT32_MAX || !isfinite(distance)) {
      handle = UINT32_MAX;
      expanded = 0;
      distance = FLT_MAX;
    }
    local_distances[item] = distance;
    local_values[item] = static_cast<u64>(handle) |
      (static_cast<u64>(expanded != 0) << 32);
  }
  __syncthreads();
  BlockSort(radix_storage).Sort(local_distances, local_values);
  for (u32 item = 0; item < ItemsPerThread; ++item) {
    const u32 output = threadIdx.x * ItemsPerThread + item;
    if (output >= beam_capacity) continue;
    beam_handles[output] = static_cast<u32>(local_values[item]);
    beam_ids[output] = UINT32_MAX;
    beam_distances[output] = local_distances[item];
    beam_expanded[output] = static_cast<u8>((local_values[item] >> 32) != 0);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    u32 valid = 0;
    const u32 limit = min(merge_count, beam_capacity);
    while (valid < limit && beam_handles[valid] != UINT32_MAX &&
           isfinite(beam_distances[valid]) && beam_distances[valid] != FLT_MAX) {
      ++valid;
    }
    beam_count = valid;
  }
  __syncthreads();
}

__device__ void merge_approximate_compact(
    u32* candidate_handles, f32* candidate_distances,
    u32* beam_handles, u32* beam_ids, f32* beam_distances,
    u8* beam_expanded, u32& beam_count, u32 beam_capacity,
    u32 existing_count, u32 merge_count,
    u32* scratch_handles, u32* scratch_expanded, f32* scratch_distances,
    CandidateWorkspace& workspace) {
  constexpr u32 pass_items =
    kApproximateSortThreadsCompact * kApproximateSortItemsCompactPass;
  for (u32 pass = 0; pass < 2; ++pass) {
    f32 local_distances[kApproximateSortItemsCompactPass];
    u64 local_values[kApproximateSortItemsCompactPass];
    for (u32 item = 0; item < kApproximateSortItemsCompactPass; ++item) {
      const u32 index = pass * pass_items +
        threadIdx.x * kApproximateSortItemsCompactPass + item;
      u32 handle = UINT32_MAX;
      u32 expanded = 0;
      f32 distance = FLT_MAX;
      if (index < existing_count) {
        handle = beam_handles[index];
        expanded = beam_expanded[index];
        distance = beam_distances[index];
      } else if (index < merge_count) {
        const u32 candidate = index - existing_count;
        handle = candidate_handles[candidate];
        distance = candidate_distances[candidate];
      }
      if (handle == UINT32_MAX || !isfinite(distance)) {
        handle = UINT32_MAX;
        expanded = 0;
        distance = FLT_MAX;
      }
      local_distances[item] = distance;
      local_values[item] = static_cast<u64>(handle) |
        (static_cast<u64>(expanded != 0) << 32);
    }
    __syncthreads();
    ApproximateBlockSortCompactPass(workspace.sort.radix_sort_compact_pass)
      .Sort(local_distances, local_values);
    for (u32 item = 0; item < kApproximateSortItemsCompactPass; ++item) {
      const u32 output =
        threadIdx.x * kApproximateSortItemsCompactPass + item;
      if (output >= beam_capacity) continue;
      const u32 destination = pass * beam_capacity + output;
      scratch_handles[destination] = static_cast<u32>(local_values[item]);
      scratch_expanded[destination] =
        static_cast<u32>((local_values[item] >> 32) != 0);
      scratch_distances[destination] = local_distances[item];
    }
    __syncthreads();
  }

  f32 final_distances[kApproximateSortItemsCompactFinal];
  u64 final_values[kApproximateSortItemsCompactFinal];
  const u32 scratch_count = beam_capacity * 2;
  for (u32 item = 0; item < kApproximateSortItemsCompactFinal; ++item) {
    const u32 index =
      threadIdx.x * kApproximateSortItemsCompactFinal + item;
    u32 handle = UINT32_MAX;
    u32 expanded = 0;
    f32 distance = FLT_MAX;
    if (index < scratch_count) {
      handle = scratch_handles[index];
      expanded = scratch_expanded[index];
      distance = scratch_distances[index];
    }
    final_distances[item] = distance;
    final_values[item] = static_cast<u64>(handle) |
      (static_cast<u64>(expanded != 0) << 32);
  }
  __syncthreads();
  ApproximateBlockSortCompactFinal(workspace.sort.radix_sort_compact_final)
    .Sort(final_distances, final_values);
  for (u32 item = 0; item < kApproximateSortItemsCompactFinal; ++item) {
    const u32 output =
      threadIdx.x * kApproximateSortItemsCompactFinal + item;
    if (output >= beam_capacity) continue;
    beam_handles[output] = static_cast<u32>(final_values[item]);
    beam_ids[output] = UINT32_MAX;
    beam_distances[output] = final_distances[item];
    beam_expanded[output] =
      static_cast<u8>((final_values[item] >> 32) != 0);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    u32 valid = 0;
    while (valid < beam_capacity && beam_handles[valid] != UINT32_MAX &&
           isfinite(beam_distances[valid]) && beam_distances[valid] != FLT_MAX) {
      ++valid;
    }
    beam_count = valid;
  }
  __syncthreads();
}

__device__ void merge_approximate_into_beam(
    u32* candidate_handles, f32* candidate_distances, u32 candidate_count,
    u32* beam_handles, u32* beam_ids, f32* beam_distances,
    u8* beam_expanded, u32& beam_count, u32 beam_capacity,
    u32* merge_handles, u32* merge_ids, f32* merge_distances,
    u8* merge_expanded, u32* compact_scratch_handles,
    u32* compact_scratch_expanded, f32* compact_scratch_distances,
    CandidateWorkspace& workspace) {
  const u32 existing_count = beam_count;
  const u32 merge_count = existing_count + candidate_count;
  if (blockDim.x != kApproximateSortThreadsWide &&
      blockDim.x != kApproximateSortThreadsCompact) {
    if (threadIdx.x == 0) beam_count = 0;
    __syncthreads();
    return;
  }
  if (blockDim.x == kApproximateSortThreadsWide) {
    merge_approximate_radix<ApproximateBlockSortWide,
                            kApproximateSortItemsWide>(
      candidate_handles, candidate_distances, candidate_count,
      beam_handles, beam_ids, beam_distances, beam_expanded,
      beam_count, beam_capacity, existing_count, merge_count,
      workspace.sort.radix_sort_wide);
  } else {
    merge_approximate_compact(
      candidate_handles, candidate_distances,
      beam_handles, beam_ids, beam_distances, beam_expanded,
      beam_count, beam_capacity, existing_count, merge_count,
      compact_scratch_handles, compact_scratch_expanded,
      compact_scratch_distances, workspace);
  }
  (void)merge_handles;
  (void)merge_ids;
  (void)merge_distances;
  (void)merge_expanded;
}

}  // namespace gpu_search::persistent_kernel_detail
