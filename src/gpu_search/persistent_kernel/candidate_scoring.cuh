#pragma once

#include "gpu_search/persistent_kernel/context.cuh"

namespace gpu_search::persistent_kernel_detail {

struct CandidateWorkspaceArrays {
  u64 handles[kPersistentMaxExact * 2];
  f32 distances[kPersistentMaxExact * 2];
  u32 ids[kPersistentMaxExact * 2];
  u8 expanded[kPersistentMaxExact * 2];
};

union CandidateSortWorkspace {
  ApproximateBlockSortWide::TempStorage radix_sort_wide;
  ApproximateBlockSortCompactPass::TempStorage radix_sort_compact_pass;
  ApproximateBlockSortCompactFinal::TempStorage radix_sort_compact_final;
  ApproximateBlockSortCompactFinal256::TempStorage
    radix_sort_compact_final_256;
};

struct CandidateWorkspace {
  CandidateWorkspaceArrays arrays;
  CandidateSortWorkspace sort;
};

inline constexpr u32 kMergeFlagExpanded = 1u;
inline constexpr u32 kMergeFlagNew = 2u;

struct FeedbackHorizonResult {
  u32 horizon{};
  u32 earliest_new_output{UINT32_MAX};
  u32 old_unexpanded_before_new{};
  u32 new_candidates_in_beam{};
};

constexpr u32 kGraphScratchBit = 0x80000000u;
constexpr u64 kNodeLockMask = 1ull;
constexpr u64 kNodeDeletedMask = 1ull << 24;
constexpr u32 kNodeIdOffset = 8;
constexpr f32 kDeviceMaxValidSquaredL2 = 0x1.fffffcp+127f;

__device__ __forceinline__ bool finite_f32_bits(f32 value) {
  return (__float_as_uint(value) & 0x7f800000u) != 0x7f800000u;
}

__device__ __forceinline__ f32 saturate_device_squared_l2(double value) {
  return value >= static_cast<double>(kDeviceMaxValidSquaredL2)
    ? kDeviceMaxValidSquaredL2 : static_cast<f32>(value);
}

__device__ __forceinline__ f32 saturate_device_component(double value) {
  constexpr double maximum = static_cast<double>(FLT_MAX);
  if (value >= maximum) return FLT_MAX;
  if (value <= -maximum) return -FLT_MAX;
  return static_cast<f32>(value);
}

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
      doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
      // Advancing only the device-side consumer index is insufficient: the
      // NIC uses the CQ doorbell record to decide whether a wrapped CQE slot
      // has been reclaimed. Without this commit every QP eventually stops
      // producing CQEs after enough successful submissions. One exclusive
      // owner consumes this CQ, so exactly one CQE is committed here. Error
      // CQEs are consumed as well before the engine enters fail-stop mode.
      doca_gpu_dev_verbs_cq_update_dbrec<false>(completion_queue, 1);
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

__device__ bool insert_visited(u64* table, u32 capacity, u64 handle) {
  const u32 mask = capacity - 1;
  u32 slot = hash64(handle) & mask;
  for (u32 probe = 0; probe < capacity; ++probe) {
    const u64 old = atomicCAS(
      reinterpret_cast<unsigned long long*>(table + slot),
      static_cast<unsigned long long>(kInvalidDeviceHandle),
      static_cast<unsigned long long>(handle));
    if (old == kInvalidDeviceHandle) return true;
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

__device__ __forceinline__ u32 remote_shard(u64 raw) {
  return static_cast<u32>((raw >> kRemoteOffsetUnitBits) & kRemoteShardMask);
}

__device__ __forceinline__ u64 remote_byte_offset(u64 raw) {
  return (raw & kRemoteOffsetUnitMask) << 4;
}

__device__ __forceinline__ u32 remote_incarnation(u64 raw) {
  return static_cast<u32>(raw >> kRemoteIncarnationShift);
}

__device__ __forceinline__ u64 make_remote_raw(u32 shard, u64 offset,
                                                u32 incarnation) {
  return (static_cast<u64>(incarnation) << kRemoteIncarnationShift) |
    (static_cast<u64>(shard) << kRemoteOffsetUnitBits) | (offset >> 4);
}

__device__ bool static_ordinal_from_raw(const PersistentKernelParams& params,
                                        u64 raw, u32& ordinal) {
  if (raw == 0 || remote_incarnation(raw) != 0) return false;
  const u32 shard = remote_shard(raw);
  const u64 offset = remote_byte_offset(raw);
  if (shard >= params.num_shards) return false;
  const DeviceShardRegion& region = params.shards[shard];
  if (offset < region.node_base_offset || region.node_stride == 0) return false;
  const u64 relative = offset - region.node_base_offset;
  if (relative % region.node_stride != 0) return false;
  const u64 slot = relative / region.node_stride;
  if (slot >= region.node_count) return false;
  ordinal = static_cast<u32>(region.ordinal_base + slot);
  return true;
}

__device__ u64 handle_from_raw(const PersistentKernelParams& params, u64 raw) {
  u32 ordinal = 0;
  if (static_ordinal_from_raw(params, raw, ordinal)) return raw;
  const u32 shard = remote_shard(raw);
  const u64 offset = remote_byte_offset(raw);
  const u32 incarnation = remote_incarnation(raw);
  if (raw == 0 || raw == kInvalidDeviceHandle ||
      incarnation == 0 || incarnation > kRemoteMaxIncarnation ||
      shard >= params.num_shards) {
    return kInvalidDeviceHandle;
  }
  const DeviceShardRegion& region = params.shards[shard];
  if (offset < region.dynamic_base_offset || region.dynamic_record_bytes == 0) {
    return kInvalidDeviceHandle;
  }
  const u64 relative = offset - region.dynamic_base_offset;
  if (relative % region.dynamic_record_bytes != 0) {
    return kInvalidDeviceHandle;
  }
  return raw;
}

__device__ bool resolve_handle(const PersistentKernelParams& params, u64 handle,
                               u64& raw, u32& shard, u64& graph_offset) {
  raw = handle;
  u32 ordinal = 0;
  if (static_ordinal_from_raw(params, raw, ordinal)) {
    u64 slot = 0;
    const DeviceShardRegion* region = shard_for_ordinal(params, ordinal, &slot);
    if (region == nullptr) return false;
    shard = region->memory_node;
    graph_offset = region->graph_base_offset +
      slot * params.graph_entry_bytes;
    return true;
  }
  if (handle_from_raw(params, raw) == kInvalidDeviceHandle) return false;
  shard = remote_shard(raw);
  const DeviceShardRegion& region = params.shards[shard];
  graph_offset = remote_byte_offset(raw) + region.dynamic_hot_offset;
  return true;
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
  if (finite_f32_bits(distance) && distance < FLT_MAX) return distance;
  double wide_distance = 0.0;
  for (u32 subquantizer = 0; subquantizer < params.pq_subquantizers;
       ++subquantizer) {
    wide_distance += static_cast<double>(
      query_lut[static_cast<size_t>(subquantizer) * 256 +
                code[subquantizer]]);
  }
  return saturate_device_squared_l2(wide_distance);
}

__device__ void beam_insert(u64* handles, u32* ids, f32* distances, u8* expanded,
                            u32& count, u32 capacity, u64 handle, u32 id, f32 distance) {
  if (handle == kInvalidDeviceHandle || !isfinite(distance) || distance == FLT_MAX) return;
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

__device__ __forceinline__ bool candidate_less(u64 lhs_handle, f32 lhs_distance,
                                               u64 rhs_handle, f32 rhs_distance) {
  return lhs_distance < rhs_distance ||
    (lhs_distance == rhs_distance && lhs_handle < rhs_handle);
}

__device__ u32 candidate_sort_capacity(u32 count) {
  u32 capacity = 1;
  while (capacity < count) capacity <<= 1;
  return capacity;
}

__device__ void sort_candidates(u64* handles, u32* ids, f32* distances,
                                u8* expanded, u32 count) {
  const u32 capacity = candidate_sort_capacity(max(1u, count));
  for (u32 index = count + threadIdx.x; index < capacity; index += blockDim.x) {
    handles[index] = kInvalidDeviceHandle;
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
        const u64 handle = handles[index];
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

__device__ __forceinline__ u32 warp_reduce_min_u32(u32 value) {
  for (u32 offset = 16; offset != 0; offset >>= 1) {
    value = min(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ __forceinline__ u32 warp_reduce_sum_u32(u32 value) {
  for (u32 offset = 16; offset != 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ void finish_feedback_horizon(
    u32 valid, u32 feedback_cap, u32 earliest_new_output,
    u32 old_unexpanded_before_new, u32 unexpanded_total,
    u32 new_candidates_in_beam, FeedbackHorizonResult* feedback) {
  if (threadIdx.x != 0 || feedback == nullptr) return;
  feedback->earliest_new_output = earliest_new_output;
  feedback->old_unexpanded_before_new = old_unexpanded_before_new;
  feedback->new_candidates_in_beam = new_candidates_in_beam;
  feedback->horizon = earliest_new_output < valid
    ? min(old_unexpanded_before_new + 1u, unexpanded_total)
    : min(unexpanded_total, feedback_cap);
}

template <class BlockSort, u32 ItemsPerThread>
__device__ void merge_approximate_radix(
    u64* candidate_handles, f32* candidate_distances, u32 candidate_count,
    u64* beam_handles, u32* beam_ids, f32* beam_distances,
    u8* beam_expanded, u32& beam_count, u32 beam_capacity,
    u32 existing_count, u32 merge_count,
    typename BlockSort::TempStorage& radix_storage,
    u32 feedback_cap, FeedbackHorizonResult* feedback) {
  __shared__ u32 earliest_new_output;
  __shared__ u32 new_candidates_in_beam;
  f32 local_distances[ItemsPerThread];
  u64 local_values[ItemsPerThread];
  u8 sorted_flags[ItemsPerThread];
  if (threadIdx.x == 0 && feedback != nullptr) {
    earliest_new_output = UINT32_MAX;
    new_candidates_in_beam = 0;
  }
  for (u32 item = 0; item < ItemsPerThread; ++item) {
    const u32 index = threadIdx.x * ItemsPerThread + item;
    u64 handle = kInvalidDeviceHandle;
    f32 distance = FLT_MAX;
    if (index < existing_count) {
      handle = beam_handles[index];
      distance = beam_distances[index];
    } else if (index < merge_count) {
      const u32 candidate = index - existing_count;
      handle = candidate_handles[candidate];
      distance = candidate_distances[candidate];
    }
    if (handle == kInvalidDeviceHandle || !isfinite(distance)) {
      handle = kInvalidDeviceHandle;
      distance = FLT_MAX;
    }
    local_distances[item] = distance;
    local_values[item] = handle;
  }
  __syncthreads();
  BlockSort(radix_storage).Sort(local_distances, local_values);
  u32 thread_earliest_new = UINT32_MAX;
  u32 thread_new_count = 0;
  for (u32 item = 0; item < ItemsPerThread; ++item) {
    u32 flags = 0;
    bool matched_old = false;
    for (u32 prior = 0; prior < existing_count; ++prior) {
      if (beam_handles[prior] == local_values[item]) {
        matched_old = true;
        flags = beam_expanded[prior] != 0 ? kMergeFlagExpanded : 0u;
        break;
      }
    }
    const u32 output = threadIdx.x * ItemsPerThread + item;
    const bool valid_new = feedback != nullptr && !matched_old &&
      output < beam_capacity &&
      local_values[item] != kInvalidDeviceHandle &&
      isfinite(local_distances[item]) && local_distances[item] != FLT_MAX;
    if (valid_new) {
      flags |= kMergeFlagNew;
      thread_earliest_new = min(thread_earliest_new, output);
      ++thread_new_count;
    }
    sorted_flags[item] = static_cast<u8>(flags);
  }
  if (feedback != nullptr) {
    const u32 lane = threadIdx.x & 31u;
    thread_earliest_new = warp_reduce_min_u32(thread_earliest_new);
    thread_new_count = warp_reduce_sum_u32(thread_new_count);
    if (lane == 0) {
      atomicMin(&earliest_new_output, thread_earliest_new);
      atomicAdd(&new_candidates_in_beam, thread_new_count);
    }
  }
  __syncthreads();
  for (u32 item = 0; item < ItemsPerThread; ++item) {
    const u32 output = threadIdx.x * ItemsPerThread + item;
    if (output >= beam_capacity) continue;
    beam_handles[output] = local_values[item];
    beam_ids[output] = UINT32_MAX;
    beam_distances[output] = local_distances[item];
    beam_expanded[output] =
      (sorted_flags[item] & kMergeFlagExpanded) != 0 ? 1u : 0u;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    u32 valid = 0;
    u32 old_unexpanded_before_new = 0;
    u32 unexpanded_total = 0;
    const u32 limit = min(merge_count, beam_capacity);
    while (valid < limit && beam_handles[valid] != kInvalidDeviceHandle &&
           isfinite(beam_distances[valid]) && beam_distances[valid] != FLT_MAX) {
      if (feedback != nullptr && beam_expanded[valid] == 0) {
        ++unexpanded_total;
        if (valid < earliest_new_output) ++old_unexpanded_before_new;
      }
      ++valid;
    }
    beam_count = valid;
    finish_feedback_horizon(
      valid, feedback_cap, earliest_new_output,
      old_unexpanded_before_new, unexpanded_total,
      new_candidates_in_beam, feedback);
  }
  __syncthreads();
}

template <u32 ItemsPerThread, class BlockSort>
__device__ void merge_approximate_compact_final(
    u64* beam_handles, u32* beam_ids, f32* beam_distances,
    u8* beam_expanded, u32& beam_count, u32 beam_capacity,
    u64* scratch_handles, u32* scratch_expanded, f32* scratch_distances,
    typename BlockSort::TempStorage& radix_storage,
    u32 feedback_cap, FeedbackHorizonResult* feedback) {
  __shared__ u32 earliest_new_output;
  __shared__ u32 new_candidates_in_beam;
  const u32 scratch_count = beam_capacity * 2;
  f32 final_distances[ItemsPerThread];
  u64 final_values[ItemsPerThread];
  u8 final_flags[ItemsPerThread];
  if (threadIdx.x == 0 && feedback != nullptr) {
    earliest_new_output = UINT32_MAX;
    new_candidates_in_beam = 0;
  }
  for (u32 item = 0; item < ItemsPerThread; ++item) {
    const u32 index = threadIdx.x * ItemsPerThread + item;
    u64 handle = kInvalidDeviceHandle;
    f32 distance = FLT_MAX;
    if (index < scratch_count) {
      handle = scratch_handles[index];
      distance = scratch_distances[index];
    }
    final_distances[item] = distance;
    final_values[item] = handle;
  }
  __syncthreads();
  BlockSort(radix_storage).Sort(final_distances, final_values);
  u32 thread_earliest_new = UINT32_MAX;
  u32 thread_new_count = 0;
  for (u32 item = 0; item < ItemsPerThread; ++item) {
    final_flags[item] = 0;
    for (u32 prior = 0; prior < scratch_count; ++prior) {
      if (scratch_handles[prior] == final_values[item]) {
        final_flags[item] = static_cast<u8>(scratch_expanded[prior]);
        break;
      }
    }
    const u32 output = threadIdx.x * ItemsPerThread + item;
    const bool valid_new = feedback != nullptr &&
      (final_flags[item] & kMergeFlagNew) != 0 &&
      output < beam_capacity &&
      final_values[item] != kInvalidDeviceHandle &&
      isfinite(final_distances[item]) && final_distances[item] != FLT_MAX;
    if (valid_new) {
      thread_earliest_new = min(thread_earliest_new, output);
      ++thread_new_count;
    }
  }
  if (feedback != nullptr) {
    const u32 lane = threadIdx.x & 31u;
    thread_earliest_new = warp_reduce_min_u32(thread_earliest_new);
    thread_new_count = warp_reduce_sum_u32(thread_new_count);
    if (lane == 0) {
      atomicMin(&earliest_new_output, thread_earliest_new);
      atomicAdd(&new_candidates_in_beam, thread_new_count);
    }
  }
  __syncthreads();
  for (u32 item = 0; item < ItemsPerThread; ++item) {
    const u32 output = threadIdx.x * ItemsPerThread + item;
    if (output >= beam_capacity) continue;
    beam_handles[output] = final_values[item];
    beam_ids[output] = UINT32_MAX;
    beam_distances[output] = final_distances[item];
    beam_expanded[output] =
      (final_flags[item] & kMergeFlagExpanded) != 0 ? 1u : 0u;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    u32 valid = 0;
    u32 old_unexpanded_before_new = 0;
    u32 unexpanded_total = 0;
    while (valid < beam_capacity &&
           beam_handles[valid] != kInvalidDeviceHandle &&
           isfinite(beam_distances[valid]) && beam_distances[valid] != FLT_MAX) {
      if (feedback != nullptr && beam_expanded[valid] == 0) {
        ++unexpanded_total;
        if (valid < earliest_new_output) ++old_unexpanded_before_new;
      }
      ++valid;
    }
    beam_count = valid;
    finish_feedback_horizon(
      valid, feedback_cap, earliest_new_output,
      old_unexpanded_before_new, unexpanded_total,
      new_candidates_in_beam, feedback);
  }
  __syncthreads();
}

__device__ void merge_approximate_compact(
    u64* candidate_handles, f32* candidate_distances,
    u64* beam_handles, u32* beam_ids, f32* beam_distances,
    u8* beam_expanded, u32& beam_count, u32 beam_capacity,
    u32 existing_count, u32 merge_count,
    u64* scratch_handles, u32* scratch_expanded, f32* scratch_distances,
    CandidateWorkspace& workspace, u32 feedback_cap = 0,
    FeedbackHorizonResult* feedback = nullptr) {
  constexpr u32 pass_items =
    kApproximateSortThreadsCompact * kApproximateSortItemsCompactPass;
  for (u32 pass = 0; pass < 2; ++pass) {
    f32 local_distances[kApproximateSortItemsCompactPass];
    u64 local_values[kApproximateSortItemsCompactPass];
    u8 sorted_flags[kApproximateSortItemsCompactPass];
    for (u32 item = 0; item < kApproximateSortItemsCompactPass; ++item) {
      const u32 index = pass * pass_items +
        threadIdx.x * kApproximateSortItemsCompactPass + item;
      u64 handle = kInvalidDeviceHandle;
      f32 distance = FLT_MAX;
      if (index < existing_count) {
        handle = beam_handles[index];
        distance = beam_distances[index];
      } else if (index < merge_count) {
        const u32 candidate = index - existing_count;
        handle = candidate_handles[candidate];
        distance = candidate_distances[candidate];
      }
      if (handle == kInvalidDeviceHandle || !isfinite(distance)) {
        handle = kInvalidDeviceHandle;
        distance = FLT_MAX;
      }
      local_distances[item] = distance;
      local_values[item] = handle;
    }
    __syncthreads();
    ApproximateBlockSortCompactPass(workspace.sort.radix_sort_compact_pass)
      .Sort(local_distances, local_values);
    for (u32 item = 0; item < kApproximateSortItemsCompactPass; ++item) {
      bool matched_old = false;
      sorted_flags[item] = 0;
      for (u32 prior = 0; prior < existing_count; ++prior) {
        if (beam_handles[prior] == local_values[item]) {
          matched_old = true;
          sorted_flags[item] = static_cast<u8>(
            beam_expanded[prior] != 0 ? kMergeFlagExpanded : 0u);
          break;
        }
      }
      if (feedback != nullptr && !matched_old &&
          local_values[item] != kInvalidDeviceHandle &&
          isfinite(local_distances[item]) &&
          local_distances[item] != FLT_MAX) {
        sorted_flags[item] |= kMergeFlagNew;
      }
    }
    __syncthreads();
    for (u32 item = 0; item < kApproximateSortItemsCompactPass; ++item) {
      const u32 output =
        threadIdx.x * kApproximateSortItemsCompactPass + item;
      if (output >= beam_capacity) continue;
      const u32 destination = pass * beam_capacity + output;
      scratch_handles[destination] = local_values[item];
      scratch_expanded[destination] = sorted_flags[item];
      scratch_distances[destination] = local_distances[item];
    }
    __syncthreads();
  }

  constexpr u32 compact_final_capacity =
    kApproximateSortThreadsCompact * kApproximateSortItemsCompactFinal;
  if (beam_capacity * 2 <= compact_final_capacity) {
    merge_approximate_compact_final<kApproximateSortItemsCompactFinal,
                                    ApproximateBlockSortCompactFinal>(
      beam_handles, beam_ids, beam_distances, beam_expanded,
      beam_count, beam_capacity, scratch_handles, scratch_expanded,
      scratch_distances, workspace.sort.radix_sort_compact_final,
      feedback_cap, feedback);
  } else {
    merge_approximate_compact_final<kApproximateSortItemsCompactFinal256,
                                    ApproximateBlockSortCompactFinal256>(
      beam_handles, beam_ids, beam_distances, beam_expanded,
      beam_count, beam_capacity, scratch_handles, scratch_expanded,
      scratch_distances, workspace.sort.radix_sort_compact_final_256,
      feedback_cap, feedback);
  }
}

__device__ void merge_approximate_into_beam(
    u64* candidate_handles, f32* candidate_distances, u32 candidate_count,
    u64* beam_handles, u32* beam_ids, f32* beam_distances,
    u8* beam_expanded, u32& beam_count, u32 beam_capacity,
    u64* merge_handles, u32* merge_ids, f32* merge_distances,
    u8* merge_expanded, u64* compact_scratch_handles,
    u32* compact_scratch_expanded, f32* compact_scratch_distances,
    CandidateWorkspace& workspace, u32 feedback_cap = 0,
    FeedbackHorizonResult* feedback = nullptr) {
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
      workspace.sort.radix_sort_wide, feedback_cap, feedback);
  } else {
    merge_approximate_compact(
      candidate_handles, candidate_distances,
      beam_handles, beam_ids, beam_distances, beam_expanded,
      beam_count, beam_capacity, existing_count, merge_count,
      compact_scratch_handles, compact_scratch_expanded,
      compact_scratch_distances, workspace, feedback_cap, feedback);
  }
  (void)merge_handles;
  (void)merge_ids;
  (void)merge_distances;
  (void)merge_expanded;
}

}  // namespace gpu_search::persistent_kernel_detail
