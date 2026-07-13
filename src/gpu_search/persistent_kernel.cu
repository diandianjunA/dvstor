#include "gpu_search/persistent_kernel.hh"

#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>

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

inline constexpr u32 kApproximateSortThreadsWide = 256;
inline constexpr u32 kApproximateSortItemsWide =
  kPersistentMaxMergeCandidates / kApproximateSortThreadsWide;
inline constexpr u32 kApproximateSortThreadsCompact = 128;
inline constexpr u32 kApproximateSortItemsCompactPass = 8;
inline constexpr u32 kApproximateSortItemsCompactFinal = 2;
using ApproximateBlockSortWide = cub::BlockRadixSort<
  f32, kApproximateSortThreadsWide, kApproximateSortItemsWide, u64>;
using ApproximateBlockSortCompactPass = cub::BlockRadixSort<
  f32, kApproximateSortThreadsCompact, kApproximateSortItemsCompactPass, u64>;
using ApproximateBlockSortCompactFinal = cub::BlockRadixSort<
  f32, kApproximateSortThreadsCompact, kApproximateSortItemsCompactFinal, u64>;

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

__device__ void direct_read_owner_loop(PersistentKernelParams params,
                                       u32 queue_count,
                                       u32 owner_block);

__global__ void persistent_search_kernel(PersistentKernelParams params) {
  const bool unified_dispatch = params.direct_owner_block_count != 0;
  if (unified_dispatch && blockIdx.x < params.direct_owner_block_count) {
    direct_read_owner_loop(params, params.direct_batch_queue_count, blockIdx.x);
    return;
  }

  bool enable_queries = true;
  bool enable_dispatcher = false;
  bool enable_delta = true;
  if (unified_dispatch) {
    const u32 role_block = blockIdx.x - params.direct_owner_block_count;
    enable_queries = role_block < params.query_block_count;
    enable_dispatcher = role_block == params.query_block_count;
    enable_delta = role_block == params.query_block_count + 1;
    if (!enable_queries && !enable_dispatcher && !enable_delta) return;
    if (threadIdx.x == 0) {
      u32* ready_count = enable_queries ? params.query_kernel_ready_count
        : enable_dispatcher ? params.dispatcher_kernel_ready_count
                           : params.control_kernel_ready_count;
      if (ready_count != nullptr) atomicAdd(ready_count, 1u);
      __threadfence_system();
    }
  } else if (threadIdx.x == 0 && params.kernel_ready_count != nullptr) {
    atomicAdd(params.kernel_ready_count, 1u);
    __threadfence_system();
  }
  __shared__ QueryDescriptor descriptor;
  __shared__ QueryDescriptor dispatch_descriptor;
  __shared__ DeltaPublishDescriptor delta_descriptor;
  __shared__ u32 have_submission;
  __shared__ u32 dispatch_pending;
  __shared__ u32 have_delta_submission;
  __shared__ u32 stop_requested;
  __shared__ u32 idle_cycles;
  __shared__ i32 delta_status;
  if (threadIdx.x == 0) {
    dispatch_pending = 0;
    idle_cycles = 256u + ((blockIdx.x * 131u) & 1023u);
  }
  __syncthreads();
  for (;;) {
    if (threadIdx.x == 0) {
      stop_requested = *reinterpret_cast<volatile u32*>(params.stop);
    }
    __syncthreads();
    if (stop_requested != 0) return;
    if (enable_dispatcher) {
      if (threadIdx.x == 0) {
        bool progressed = false;
        if (dispatch_pending == 0 && params.submissions.entries != nullptr &&
            device_ring_try_pop(params.submissions, dispatch_descriptor)) {
          dispatch_pending = 1;
          progressed = true;
        }
        if (dispatch_pending != 0 &&
            params.device_submissions.entries != nullptr &&
            device_ring_try_push(params.device_submissions,
                                 dispatch_descriptor)) {
          dispatch_pending = 0;
          progressed = true;
        }
        if (progressed) {
          idle_cycles = 256u + ((blockIdx.x * 131u) & 1023u);
        } else {
          device_ring_relax(idle_cycles);
          idle_cycles = min(idle_cycles * 2, 16384u);
        }
      }
      __syncthreads();
      continue;
    }
    if (threadIdx.x == 0) {
      have_delta_submission = enable_delta &&
        params.delta_submissions.entries != nullptr &&
        device_ring_try_pop(params.delta_submissions, delta_descriptor) ? 1u : 0u;
    }
    __syncthreads();
    if (have_delta_submission != 0) {
      if (threadIdx.x == 0) {
        delta_status = 0;
        const bool reset = (delta_descriptor.flags & kDeltaCommandReset) != 0;
        const bool promote =
          (delta_descriptor.flags & kDeltaCommandPromoteOverrides) != 0;
        constexpr u32 known_flags =
          kDeltaCommandReset | kDeltaCommandPromoteOverrides;
        if ((delta_descriptor.flags & ~known_flags) != 0 ||
            (reset && promote) ||
            (reset && (delta_descriptor.first_slot != 0 ||
                       delta_descriptor.record_count > params.delta_capacity ||
                       delta_descriptor.final_count != 0 ||
                       delta_descriptor.invalidation_count != 0 ||
                       delta_descriptor.superseded_count != 0 ||
                       delta_descriptor.override_count != 0 ||
                       delta_descriptor.durable_count != 0 ||
                       delta_descriptor.resident_pq_erase_count != 0 ||
                       params.delta_records == nullptr ||
                       params.delta_next == nullptr ||
                       params.delta_prev == nullptr ||
                       params.delta_remote_positions == nullptr ||
                       params.delta_count == nullptr ||
                       params.base_override_keys == nullptr ||
                       params.base_override_epochs == nullptr ||
                       params.base_override_capacity == 0 ||
                       params.delta_remote_keys == nullptr ||
                       params.delta_remote_slots == nullptr ||
                       params.delta_remote_capacity == 0 ||
                       (params.anchor_count != 0 &&
                        params.delta_bucket_heads == nullptr))) ||
            (!reset && (delta_descriptor.final_count > params.delta_capacity ||
            delta_descriptor.record_count > params.delta_capacity ||
            (delta_descriptor.record_count != 0 &&
             (params.delta_staging_slots == nullptr ||
              params.delta_staging_records == nullptr)) ||
            (delta_descriptor.record_count != 0 &&
             (params.delta_remote_positions == nullptr ||
              params.delta_remote_capacity == 0)) ||
            (delta_descriptor.record_count != 0 && params.vector_bytes != 0 &&
             (params.delta_staging_vectors == nullptr || params.delta_vectors == nullptr)) ||
            (delta_descriptor.record_count != 0 && params.pq_code_bytes != 0 &&
             (params.delta_pq_codes == nullptr || params.delta_encode_scratch == nullptr ||
              params.pq_centroids == nullptr || params.resident_pq_codes == nullptr ||
              params.resident_pq_keys == nullptr ||
              params.resident_pq_slots == nullptr ||
              params.resident_pq_positions == nullptr ||
              params.resident_pq_capacity == 0 ||
              params.resident_pq_table_capacity == 0)) ||
            (delta_descriptor.durable_count != 0 &&
             params.delta_durable_updates == nullptr) ||
            (delta_descriptor.resident_pq_erase_count != 0 &&
             (params.resident_pq_erase_updates == nullptr ||
              params.resident_pq_keys == nullptr ||
              params.resident_pq_slots == nullptr ||
              params.resident_pq_positions == nullptr ||
              params.resident_pq_capacity == 0 ||
              params.resident_pq_table_capacity == 0)) ||
            (promote && delta_descriptor.override_count != 0 &&
             (params.delta_override_updates == nullptr ||
              params.permanent_override_bits == nullptr ||
              params.permanent_override_words == 0)) ||
            (!promote && delta_descriptor.override_count != 0 &&
             (params.delta_override_updates == nullptr ||
              params.base_override_keys == nullptr ||
              params.base_override_epochs == nullptr ||
              params.base_override_capacity == 0))))) {
          delta_status = -EINVAL;
        }
      }
      __syncthreads();

      if ((delta_descriptor.flags & kDeltaCommandReset) != 0) {
        if (delta_status == 0) {
          for (u32 index = threadIdx.x; index < delta_descriptor.record_count;
               index += blockDim.x) {
            const DeviceDeltaRecord record = params.delta_records[index];
            const u32 remote_position = params.delta_remote_positions[index];
            if (record.remote_node != 0 &&
                remote_position < params.delta_remote_capacity &&
                load_cg(params.delta_remote_slots + remote_position) == index) {
              atomicCAS(reinterpret_cast<unsigned long long*>(
                          params.delta_remote_keys + remote_position),
                        record.remote_node, kDeltaRemoteTombstone);
              atomicExch(params.delta_remote_slots + remote_position, UINT32_MAX);
            }
            if (record.base_ordinal < params.num_nodes) {
              const u32 mask = params.base_override_capacity - 1;
              u32 position = hash32(record.base_ordinal) & mask;
              for (u32 probe = 0; probe < params.base_override_capacity; ++probe) {
                const u32 key = load_cg(params.base_override_keys + position);
                if (key == record.base_ordinal) {
                  if (atomicCAS(params.base_override_keys + position,
                                record.base_ordinal,
                                kBaseOverrideTombstone) == record.base_ordinal) {
                    params.base_override_epochs[position] = 0;
                  }
                  break;
                }
                if (key == kBaseOverrideEmpty) break;
                position = (position + 1) & mask;
              }
            }
            params.delta_records[index] = {};
            params.delta_records[index].base_ordinal = kBaseOverrideEmpty;
            params.delta_next[index] = UINT32_MAX;
            params.delta_prev[index] = UINT32_MAX;
            params.delta_remote_positions[index] = UINT32_MAX;
          }
          for (u32 index = threadIdx.x; index < params.anchor_count;
               index += blockDim.x) {
            params.delta_bucket_heads[index] = UINT32_MAX;
          }
        }
        __syncthreads();
        if (threadIdx.x == 0 && delta_status == 0) {
          __threadfence();
          atomicExch(params.delta_count, 0u);
          __threadfence_system();
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          device_ring_push(params.delta_completions, DeltaPublishCompletion{
            .command_id = delta_descriptor.command_id,
            .status = delta_status,
            .final_count = 0,
          });
        }
        __syncthreads();
        continue;
      }

      if (delta_status == 0) {
        for (u32 index = threadIdx.x;
             index < delta_descriptor.record_count;
             index += blockDim.x) {
          const u32 slot = params.delta_staging_slots[index];
          if (slot >= delta_descriptor.final_count || slot >= params.delta_capacity) {
            atomicExch(&delta_status, -EINVAL);
          }
        }
        __syncthreads();
      }

      if (delta_status == 0) {
        for (u32 index = threadIdx.x;
             index < delta_descriptor.record_count;
             index += blockDim.x) {
          const u32 slot = params.delta_staging_slots[index];
          params.delta_records[slot] = params.delta_staging_records[index];
          params.delta_next[slot] = UINT32_MAX;
          params.delta_prev[slot] = UINT32_MAX;
        }
        for (u64 index = threadIdx.x;
             index < static_cast<u64>(delta_descriptor.record_count) * params.vector_bytes;
             index += blockDim.x) {
          const u32 record_index = static_cast<u32>(index / params.vector_bytes);
          const u32 byte = static_cast<u32>(index % params.vector_bytes);
          const u32 slot = params.delta_staging_slots[record_index];
          params.delta_vectors[static_cast<u64>(slot) * params.vector_bytes + byte] =
            params.delta_staging_vectors[index];
        }
        __syncthreads();

        for (u64 index = threadIdx.x;
             index < static_cast<u64>(delta_descriptor.record_count) * params.dim;
             index += blockDim.x) {
          const u32 record_index = static_cast<u32>(index / params.dim);
          const u32 row = static_cast<u32>(index % params.dim);
          const u32 slot = params.delta_staging_slots[record_index];
          const DeviceDeltaRecord record = params.delta_records[slot];
          f32 transformed = 0.0f;
          if ((record.flags & kDeltaDeleted) == 0) {
            const u8* vector = params.delta_vectors +
              static_cast<size_t>(slot) * params.vector_bytes;
            if (params.opq_matrix == nullptr) {
              transformed = storage_component(params, vector, row);
            } else {
              const f32* matrix_row = params.opq_matrix +
                static_cast<size_t>(row) * params.dim;
              for (u32 column = 0; column < params.dim; ++column) {
                transformed += matrix_row[column] *
                  storage_component(params, vector, column);
              }
            }
          }
          params.delta_encode_scratch[index] = transformed;
        }
        __syncthreads();

        for (u64 index = threadIdx.x;
             index < static_cast<u64>(delta_descriptor.record_count) * params.pq_code_bytes;
             index += blockDim.x) {
          const u32 record_index = static_cast<u32>(index / params.pq_code_bytes);
          const u32 subquantizer = static_cast<u32>(index % params.pq_code_bytes);
          const u32 slot = params.delta_staging_slots[record_index];
          u8 best_code = 0;
          if ((params.delta_records[slot].flags & kDeltaDeleted) == 0) {
            const f32* transformed = params.delta_encode_scratch +
              static_cast<size_t>(record_index) * params.dim +
              static_cast<size_t>(subquantizer) * params.pq_subvector_dim;
            const f32* centroids = params.pq_centroids +
              static_cast<size_t>(subquantizer) * 256 * params.pq_subvector_dim;
            f32 best_distance = FLT_MAX;
            for (u32 centroid = 0; centroid < 256; ++centroid) {
              f32 distance = 0.0f;
              for (u32 dimension = 0; dimension < params.pq_subvector_dim; ++dimension) {
                const f32 difference = transformed[dimension] -
                  centroids[static_cast<size_t>(centroid) * params.pq_subvector_dim + dimension];
                distance += difference * difference;
              }
              if (distance < best_distance) {
                best_distance = distance;
                best_code = static_cast<u8>(centroid);
              }
            }
          }
          params.delta_pq_codes[
            static_cast<size_t>(slot) * params.pq_code_bytes + subquantizer] = best_code;
          const u32 resident_slot = params.delta_records[slot].resident_pq_slot;
          if ((params.delta_records[slot].flags & kDeltaDeleted) == 0) {
            if (resident_slot >= params.resident_pq_capacity) {
              atomicExch(&delta_status, -ENOSPC);
            } else {
              params.resident_pq_codes[
                static_cast<size_t>(resident_slot) * params.pq_code_bytes +
                subquantizer] = best_code;
            }
          }
        }
        __threadfence();
        __syncthreads();

        for (u32 index = threadIdx.x;
             index < delta_descriptor.invalidation_count;
             index += blockDim.x) {
          const u64 key = params.graph_invalidation_keys[index];
          const u32 route_slot = anchor_graph_slot(params, key);
          if (route_slot != UINT32_MAX &&
              params.anchor_graph_states != nullptr) {
            atomicCAS(params.anchor_graph_states + route_slot,
                      kGraphCacheReady, kGraphCacheStale);
          }
          if (params.graph_cache_sets == 0 ||
              params.graph_cache_states == nullptr ||
              params.graph_cache_keys == nullptr) {
            continue;
          }
          const u32 set = hash64(key) % params.graph_cache_sets;
          for (u32 way = 0; way < params.graph_cache_ways; ++way) {
            const u32 slot = set * params.graph_cache_ways + way;
            for (;;) {
              const u32 state = *reinterpret_cast<volatile u32*>(
                params.graph_cache_states + slot);
              if (load_cg(params.graph_cache_keys + slot) != key ||
                  state == kGraphCacheEmpty || state == kGraphCacheStale ||
                  state == kGraphCacheFillInvalidated) {
                break;
              }
              if (state == kGraphCacheReady) {
                if (atomicCAS(params.graph_cache_states + slot, kGraphCacheReady,
                              kGraphCacheStale) == kGraphCacheReady) {
                  break;
                }
                continue;
              }
              if (state == kGraphCacheFilling) {
                if (atomicCAS(params.graph_cache_states + slot, kGraphCacheFilling,
                              kGraphCacheFillInvalidated) == kGraphCacheFilling) {
                  break;
                }
                continue;
              }
              break;
            }
          }
        }

        if (threadIdx.x == 0) {
          for (u32 index = 0; index < delta_descriptor.superseded_count; ++index) {
            const DeltaSupersedeUpdate update = params.delta_supersede_updates[index];
            if (update.slot >= delta_descriptor.final_count) {
              delta_status = -EINVAL;
              continue;
            }
            DeviceDeltaRecord& record = params.delta_records[update.slot];
            record.superseded_epoch = update.epoch;
            unlink_mutable_delta(params, update.slot);
          }
        }

        if ((delta_descriptor.flags & kDeltaCommandPromoteOverrides) != 0) {
          for (u32 index = threadIdx.x;
               index < delta_descriptor.override_count;
               index += blockDim.x) {
            const u32 ordinal = params.delta_override_updates[index].ordinal;
            if (ordinal >= params.num_nodes) {
              atomicExch(&delta_status, -EINVAL);
              continue;
            }
            atomicOr(params.permanent_override_bits + ordinal / 32,
                     1u << (ordinal % 32));
          }
        } else if (threadIdx.x == 0) {
          const u32 mask = params.base_override_capacity - 1;
          for (u32 index = 0; index < delta_descriptor.override_count; ++index) {
            const DeltaOverrideUpdate update = params.delta_override_updates[index];
            u32 position = hash32(update.ordinal) & mask;
            u32 first_tombstone = UINT32_MAX;
            bool inserted = false;
            for (u32 probe = 0; probe < params.base_override_capacity; ++probe) {
              const u32 key = params.base_override_keys[position];
              if (key == update.ordinal) {
                params.base_override_epochs[position] = min(
                  params.base_override_epochs[position], update.epoch);
                inserted = true;
                break;
              }
              if (key == kBaseOverrideTombstone && first_tombstone == UINT32_MAX) {
                first_tombstone = position;
              }
              if (key == kBaseOverrideEmpty) {
                const u32 destination = first_tombstone == UINT32_MAX
                  ? position : first_tombstone;
                params.base_override_epochs[destination] = update.epoch;
                __threadfence();
                params.base_override_keys[destination] = update.ordinal;
                inserted = true;
                break;
              }
              position = (position + 1) & mask;
            }
            if (!inserted && first_tombstone != UINT32_MAX) {
              params.base_override_epochs[first_tombstone] = update.epoch;
              __threadfence();
              params.base_override_keys[first_tombstone] = update.ordinal;
              inserted = true;
            }
            if (!inserted) delta_status = -ENOSPC;
          }
        }

        if (threadIdx.x == 0) {
          for (u32 index = 0; index < delta_descriptor.durable_count; ++index) {
            const DeltaDurableUpdate update = params.delta_durable_updates[index];
            if (update.slot >= delta_descriptor.final_count) {
              delta_status = -EINVAL;
              continue;
            }
            DeviceDeltaRecord& record = params.delta_records[update.slot];
            if (record.epoch == update.epoch) {
              if (record.superseded_epoch == 0) {
                record.superseded_epoch = update.epoch;
              }
              unlink_mutable_delta(params, update.slot);
              const u32 remote_position = params.delta_remote_positions[update.slot];
              if (record.remote_node != 0 &&
                  remote_position < params.delta_remote_capacity &&
                  load_cg(params.delta_remote_slots + remote_position) == update.slot) {
                atomicCAS(reinterpret_cast<unsigned long long*>(
                            params.delta_remote_keys + remote_position),
                          record.remote_node, kDeltaRemoteTombstone);
                atomicExch(params.delta_remote_slots + remote_position, UINT32_MAX);
              }
              params.delta_remote_positions[update.slot] = UINT32_MAX;
              if (record.base_ordinal < params.num_nodes) {
                atomicOr(params.permanent_override_bits + record.base_ordinal / 32,
                         1u << (record.base_ordinal % 32));
                const u32 mask = params.base_override_capacity - 1;
                u32 position = hash32(record.base_ordinal) & mask;
                for (u32 probe = 0; probe < params.base_override_capacity; ++probe) {
                  const u32 key = load_cg(params.base_override_keys + position);
                  if (key == record.base_ordinal) {
                    if (atomicCAS(params.base_override_keys + position,
                                  record.base_ordinal,
                                  kBaseOverrideTombstone) == record.base_ordinal) {
                      params.base_override_epochs[position] = 0;
                    }
                    break;
                  }
                  if (key == kBaseOverrideEmpty) break;
                  position = (position + 1) & mask;
                }
              }
            }
          }
          for (u32 index = 0;
               index < delta_descriptor.resident_pq_erase_count; ++index) {
            erase_resident_pq(params, params.resident_pq_erase_updates[index]);
          }
        }
      }
      __syncthreads();

      if (delta_status == 0) {
        if (threadIdx.x == 0) {
          const u32 mask = params.delta_remote_capacity - 1;
          for (u32 index = 0; index < delta_descriptor.record_count; ++index) {
            const u32 slot = params.delta_staging_slots[index];
            const DeviceDeltaRecord record = params.delta_records[slot];
            if ((record.flags & kDeltaDeleted) == 0 &&
                !insert_resident_pq(
                  params, record.remote_node, record.resident_pq_slot)) {
              delta_status = -ENOSPC;
              break;
            }
            params.delta_remote_positions[slot] = UINT32_MAX;
            if (record.remote_node != 0 && params.delta_remote_capacity != 0) {
              u32 position = hash64(record.remote_node) & mask;
              u32 first_tombstone = UINT32_MAX;
              bool inserted = false;
              for (u32 probe = 0; probe < params.delta_remote_capacity; ++probe) {
                const u64 key = params.delta_remote_keys[position];
                if (key == record.remote_node) {
                  params.delta_remote_slots[position] = slot;
                  params.delta_remote_positions[slot] = position;
                  inserted = true;
                  break;
                }
                if (key == kDeltaRemoteTombstone && first_tombstone == UINT32_MAX) {
                  first_tombstone = position;
                }
                if (key == kDeltaRemoteEmpty) {
                  const u32 destination = first_tombstone == UINT32_MAX
                    ? position : first_tombstone;
                  params.delta_remote_slots[destination] = slot;
                  __threadfence();
                  params.delta_remote_keys[destination] = record.remote_node;
                  params.delta_remote_positions[slot] = destination;
                  inserted = true;
                  break;
                }
                position = (position + 1) & mask;
              }
              if (!inserted && first_tombstone != UINT32_MAX) {
                params.delta_remote_slots[first_tombstone] = slot;
                __threadfence();
                params.delta_remote_keys[first_tombstone] = record.remote_node;
                params.delta_remote_positions[slot] = first_tombstone;
                inserted = true;
              }
              if (!inserted) {
                delta_status = -ENOSPC;
                break;
              }
            }
            if ((record.flags & (kDeltaDeleted | kDeltaDurable)) == 0 &&
                record.superseded_epoch == 0 &&
                params.delta_bucket_heads != nullptr) {
              const u32 old_head = params.delta_bucket_heads[record.anchor_bucket];
              params.delta_prev[slot] = UINT32_MAX;
              params.delta_next[slot] = old_head;
              if (old_head < params.delta_capacity) {
                params.delta_prev[old_head] = slot;
              }
              params.delta_bucket_heads[record.anchor_bucket] = slot;
            }
          }
        }
      }
      __syncthreads();
      if (threadIdx.x == 0 && delta_status == 0) {
        __threadfence();
        atomicExch(params.delta_count, delta_descriptor.final_count);
        __threadfence_system();
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        device_ring_push(params.delta_completions, DeltaPublishCompletion{
          .command_id = delta_descriptor.command_id,
          .status = delta_status,
          .final_count = delta_status == 0 ? delta_descriptor.final_count : 0u,
        });
      }
      __syncthreads();
      if (threadIdx.x == 0) idle_cycles = 256u;
      __syncthreads();
      continue;
    }

    if (threadIdx.x == 0) {
      const DeviceRingView<QueryDescriptor> query_queue =
        params.device_submissions.entries != nullptr
          ? params.device_submissions : params.submissions;
      have_submission = enable_queries && query_queue.entries != nullptr &&
        device_ring_try_pop(query_queue, descriptor) ? 1u : 0u;
    }
    __syncthreads();
    if (have_submission == 0) {
      if (threadIdx.x == 0) {
        device_ring_relax(idle_cycles);
        idle_cycles = min(idle_cycles * 2, 16384u);
      }
      __syncthreads();
      continue;
    }
    if (threadIdx.x == 0) {
      idle_cycles = 256u + ((blockIdx.x * 131u) & 1023u);
    }
    __syncthreads();
    process_query(params, descriptor);
    __syncthreads();
  }
}

__device__ void complete_direct_batch(const DirectBatchDescriptor& descriptor,
                                      i32 status) {
  if (descriptor.completion_status == nullptr) return;
  __threadfence_system();
  atomicExch(descriptor.completion_status, status);
}

__device__ void direct_read_owner_loop(PersistentKernelParams params,
                                       u32 queue_count,
                                       u32 owner_block) {
#ifdef DVSTOR_HAVE_GPUNETIO
  constexpr u32 warp_width = 32;
  constexpr u32 max_warps_per_block = 8;
  constexpr u32 max_submit_batches = 8;
  const u32 lane = threadIdx.x % warp_width;
  const u32 warps_per_block = blockDim.x / warp_width;
  const u32 warp_in_block = threadIdx.x / warp_width;
  const u32 warp = owner_block * warps_per_block + warp_in_block;
  if (warps_per_block == 0 || warps_per_block > max_warps_per_block ||
      warp >= queue_count) return;
  if (lane == 0 && params.direct_owner_phases != nullptr) {
    params.direct_owner_phases[warp] = 10;
    __threadfence_system();
  }
  u32 invalid = 0;
  invalid |= params.direct_batch_queues == nullptr ? 1u : 0u;
  invalid |= params.direct_qps == nullptr ? 2u : 0u;
  invalid |= params.direct_regions == nullptr ? 4u : 0u;
  invalid |= params.direct_disabled == nullptr ? 8u : 0u;
  invalid |= params.direct_region_count == 0 ? 16u : 0u;
  invalid |= params.direct_qps_per_node == 0 ? 32u : 0u;
  invalid |= warp >= params.direct_batch_queue_count ? 64u : 0u;
  if (invalid != 0) {
    if (lane == 0 && params.direct_owner_phases != nullptr) {
      params.direct_owner_phases[warp] = 0x100u | invalid;
      __threadfence_system();
    }
    return;
  }

  __shared__ DirectBatchDescriptor shared_batches
    [max_warps_per_block][max_submit_batches];
  __shared__ u32 shared_matching_counts
    [max_warps_per_block][max_submit_batches];
  __shared__ u32 shared_wqe_offsets
    [max_warps_per_block][max_submit_batches];
  __shared__ u32 shared_batch_counts[max_warps_per_block];
  __shared__ u32 shared_total_wqes[max_warps_per_block];

  const u32 memory_node = warp % params.direct_region_count;
  auto* qp = reinterpret_cast<doca_gpu_dev_verbs_qp*>(params.direct_qps[warp]);
  if (qp == nullptr) {
    if (lane == 0 && params.direct_owner_phases != nullptr) {
      params.direct_owner_phases[warp] = 0x200u;
      __threadfence_system();
    }
    return;
  }
  auto* completion_queue = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
  const bool need_dump = qp->need_dump;
  const DirectRemoteRegion& region = params.direct_regions[memory_node];
  const DeviceRingView<DirectBatchDescriptor> queue =
    params.direct_batch_queues[warp];
  DirectBatchDescriptor deferred{};
  u32 deferred_matching = 0;
  bool have_deferred = false;
  bool trace_first_batch = true;

  if (lane == 0 && params.direct_owner_phases != nullptr) {
    params.direct_owner_phases[warp] = 1;
    __threadfence_system();
  }

  const u32 initial_idle_cycles = 256u + ((warp * 97u) & 2047u);
  u32 idle_cycles = initial_idle_cycles;
  for (;;) {
    const u32 stop_requested = lane == 0
      ? *reinterpret_cast<const volatile u32*>(params.stop)
      : 0u;
    if (__shfl_sync(0xffffffffu, stop_requested, 0) != 0) break;

    if (lane == 0) {
      u32 batch_count = 0;
      u32 total_wqes = 0;
      while (batch_count < max_submit_batches) {
        DirectBatchDescriptor descriptor{};
        u32 matching = 0;
        if (have_deferred) {
          descriptor = deferred;
          matching = deferred_matching;
          have_deferred = false;
        } else if (!device_ring_try_pop(queue, descriptor)) {
          break;
        } else {
          if (trace_first_batch && params.direct_owner_phases != nullptr) {
            params.direct_owner_phases[warp] = 2;
            __threadfence_system();
          }
          if (descriptor.memory_node == memory_node &&
              descriptor.request_shards != nullptr &&
              descriptor.remote_offsets != nullptr &&
              descriptor.local_iova_offsets != nullptr &&
              descriptor.bytes != 0 &&
              descriptor.bytes <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
            for (u32 index = 0; index < descriptor.request_count; ++index) {
              matching += descriptor.request_shards[index] == memory_node ? 1u : 0u;
            }
          }
        }

        if (descriptor.memory_node != memory_node || matching == 0 ||
            descriptor.bytes == 0 ||
            descriptor.bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
          complete_direct_batch(descriptor, -EINVAL);
          continue;
        }
        if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) != 0) {
          complete_direct_batch(descriptor, -EHOSTDOWN);
          continue;
        }
        const u32 needed = matching + (need_dump ? 1u : 0u);
        if (needed > qp->sq_wqe_num) {
          complete_direct_batch(descriptor, -E2BIG);
          continue;
        }
        if (batch_count != 0 && total_wqes + needed > qp->sq_wqe_num) {
          deferred = descriptor;
          deferred_matching = matching;
          have_deferred = true;
          break;
        }
        shared_batches[warp_in_block][batch_count] = descriptor;
        shared_matching_counts[warp_in_block][batch_count] = matching;
        shared_wqe_offsets[warp_in_block][batch_count] = total_wqes;
        ++batch_count;
        total_wqes += needed;
      }
      shared_batch_counts[warp_in_block] = batch_count;
      shared_total_wqes[warp_in_block] = total_wqes;
    }
    __syncwarp();

    const u32 batch_count = shared_batch_counts[warp_in_block];
    if (batch_count == 0) {
      if (lane == 0) device_ring_relax(idle_cycles);
      __syncwarp();
      idle_cycles = min(idle_cycles * 2, 16384u);
      continue;
    }
    idle_cycles = initial_idle_cycles;

    const doca_gpu_dev_verbs_ticket_t first_wqe = qp->sq_wqe_pi;
    const doca_gpu_dev_verbs_ticket_t first_completion =
      doca_gpu_dev_verbs_load_relaxed<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
          &completion_queue->cqe_ci);
    for (u32 batch = 0; batch < batch_count; ++batch) {
      const DirectBatchDescriptor descriptor =
        shared_batches[warp_in_block][batch];
      const u32 matching = shared_matching_counts[warp_in_block][batch];
      const u32 batch_offset = shared_wqe_offsets[warp_in_block][batch];
      u32 matched_before = 0;
      for (u32 base = 0; base < descriptor.request_count; base += warp_width) {
        const u32 index = base + lane;
        const bool matching_request = index < descriptor.request_count &&
          descriptor.request_shards[index] == memory_node;
        const u32 matching_mask = __ballot_sync(0xffffffffu, matching_request);
        if (matching_request) {
          const u32 lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
          const u32 rank = __popc(matching_mask & lower_lanes);
          const u32 matched = matched_before + rank;
          const doca_gpu_dev_verbs_ticket_t ticket =
            first_wqe + batch_offset + matched;
          auto* wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, ticket);
          const bool last_read = matched + 1 == matching;
          const auto flags = !need_dump && last_read
            ? DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE
            : DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;
          doca_gpu_dev_verbs_wqe_prepare_read(
            qp, wqe, ticket, flags,
            region.address + descriptor.remote_offsets[index], region.rkey,
            descriptor.local_iova_offsets[index], params.direct_local_mkey,
            descriptor.bytes);
        }
        matched_before += __popc(matching_mask);
      }
      if (need_dump && lane == 0) {
        const doca_gpu_dev_verbs_ticket_t ticket =
          first_wqe + batch_offset + matching;
        auto* dump_wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, ticket);
        doca_gpu_dev_verbs_wqe_prepare_dump(
          qp, dump_wqe, ticket, DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
          reinterpret_cast<u64>(params.direct_dump) - params.direct_local_iova_base,
          params.direct_local_mkey, 1);
      }
    }
    __syncwarp();
    if (lane == 0) {
      if (trace_first_batch && params.direct_owner_phases != nullptr) {
        params.direct_owner_phases[warp] = 3;
        __threadfence_system();
      }
      doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
        qp, first_wqe + shared_total_wqes[warp_in_block]);
      if (trace_first_batch && params.direct_owner_phases != nullptr) {
        params.direct_owner_phases[warp] = 4;
        __threadfence_system();
      }

      i32 status = 0;
      for (u32 batch = 0; batch < batch_count; ++batch) {
        if (status == 0) {
          status = poll_direct_cq(completion_queue, first_completion + batch,
                                  params.direct_timeout_ns, params.stop,
                                  params.direct_disabled);
        }
        complete_direct_batch(shared_batches[warp_in_block][batch], status);
      }
      if (trace_first_batch && params.direct_owner_phases != nullptr) {
        params.direct_owner_phases[warp] = status == 0 ? 6u : 5u;
        __threadfence_system();
        trace_first_batch = false;
      }
      if (status != 0 && status != -ECANCELED) {
        if (params.direct_error != nullptr) atomicCAS(params.direct_error, 0, status);
        atomicExch(params.direct_disabled, 1u);
      }
    }
    __syncwarp();
  }
#else
  (void)params;
  (void)queue_count;
  (void)owner_block;
#endif
}

__global__ void direct_read_owner_kernel(PersistentKernelParams params,
                                         u32 queue_count) {
  direct_read_owner_loop(params, queue_count, blockIdx.x);
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

__global__ void gpunetio_owner_read_probe_kernel(
    PersistentKernelParams params, u32* request_shards,
    u64* remote_offsets, u64* local_iova_offsets, u8* destinations,
    u32 destination_stride, i32* statuses, u32* completed,
    u32* phases, u32 queue_count) {
  const u32 qp_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (qp_index >= queue_count || params.direct_region_count == 0) return;
  phases[qp_index] = 1;
  __threadfence_system();
  const u32 memory_node = qp_index % params.direct_region_count;
  const u32 lane = qp_index / params.direct_region_count;
  request_shards[qp_index] = memory_node;
  remote_offsets[qp_index] = 0;
  local_iova_offsets[qp_index] =
    reinterpret_cast<u64>(destinations +
      static_cast<size_t>(qp_index) * destination_stride) -
    params.direct_local_iova_base;
  __threadfence();
  i32* completion_status = statuses + qp_index;
  const i32 status = direct_fetch_batch(
    params, memory_node, request_shards + qp_index,
    remote_offsets + qp_index, 1,
    destinations + static_cast<size_t>(qp_index) * destination_stride,
    destination_stride, sizeof(u64), lane,
    local_iova_offsets + qp_index, completion_status, false,
    phases + qp_index);
  statuses[qp_index] = status;
  if (status == 0) atomicAdd(completed, 1u);
  __threadfence_system();
  phases[qp_index] = 4;
  __threadfence_system();
}

__global__ void gather_anchor_codes_kernel(const u8* base_codes,
                                           const u32* anchor_handles,
                                           u8* anchor_codes,
                                           u32 anchor_count,
                                           u32 code_bytes,
                                           u32 node_count) {
  const u64 byte = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
  const u64 total = static_cast<u64>(anchor_count) * code_bytes;
  if (byte >= total) return;
  const u32 anchor = static_cast<u32>(byte / code_bytes);
  const u32 code_byte = static_cast<u32>(byte % code_bytes);
  const u32 handle = anchor_handles[anchor];
  anchor_codes[byte] = handle < node_count
    ? base_codes[static_cast<u64>(handle) * code_bytes + code_byte]
    : 0;
}

}  // namespace

void launch_persistent_search(cudaStream_t stream, const PersistentKernelParams& params,
                              u32 blocks, u32 threads) {
  persistent_search_kernel<<<blocks, threads, 0, stream>>>(params);
}

void launch_direct_read_owners(cudaStream_t stream,
                               const PersistentKernelParams& params,
                               u32 queue_count, u32 threads) {
  const u32 warps_per_block = max(1u, threads / 32);
  const u32 blocks = (queue_count + warps_per_block - 1) / warps_per_block;
  direct_read_owner_kernel<<<blocks, threads, 0, stream>>>(params, queue_count);
}

void launch_gpunetio_owner_read_probe(
    cudaStream_t stream, const PersistentKernelParams& params,
    u32* request_shards, u64* remote_offsets, u64* local_iova_offsets,
    u8* destinations, u32 destination_stride, i32* statuses,
    u32* completed, u32* phases, u32 queue_count) {
  constexpr u32 threads = 128;
  const u32 blocks = (queue_count + threads - 1) / threads;
  gpunetio_owner_read_probe_kernel<<<blocks, threads, 0, stream>>>(
    params, request_shards, remote_offsets, local_iova_offsets,
    destinations, destination_stride, statuses, completed, phases, queue_count);
}

void launch_gather_anchor_codes(cudaStream_t stream, const u8* base_codes,
                                const u32* anchor_handles, u8* anchor_codes,
                                u32 anchor_count, u32 code_bytes,
                                u32 node_count) {
  const u64 bytes = static_cast<u64>(anchor_count) * code_bytes;
  if (bytes == 0) return;
  constexpr u32 threads = 256;
  const u32 blocks = static_cast<u32>((bytes + threads - 1) / threads);
  gather_anchor_codes_kernel<<<blocks, threads, 0, stream>>>(
    base_codes, anchor_handles, anchor_codes, anchor_count, code_bytes, node_count);
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

}  // namespace gpu_search
