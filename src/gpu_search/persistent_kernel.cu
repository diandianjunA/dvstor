#include "gpu_search/persistent_kernel.hh"

#include <cuda_runtime.h>

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

__device__ void beam_insert(u32* handles, f32* distances, u8* expanded,
                            u32& count, u32 capacity, u32 handle, f32 distance) {
  if (handle == UINT32_MAX || !isfinite(distance) || distance == FLT_MAX) return;
  if (count < capacity) {
    handles[count] = handle;
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
      params.direct_qps_per_node == 0) return -EINVAL;
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
  doca_gpu_dev_verbs_get<DOCA_GPUNETIO_VERBS_NODUMP,
                         DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                         DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(
    qp, remote, local, bytes, dump, &ticket);
  return doca_gpu_dev_verbs_poll_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
    doca_gpu_dev_verbs_qp_get_cq_sq(qp), ticket);
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
        params.fetch_status[status_index] = 0;
        if (params.direct_backend != 0) {
          const i32 status = direct_fetch(params, shard, graph_offset, destination,
                                          params.graph_entry_bytes, request_lane + retry);
          params.fetch_status[status_index] = status == 0 ? 1 : status;
        } else {
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
          while (*reinterpret_cast<volatile i32*>(params.fetch_status + status_index) == 0 &&
                 *reinterpret_cast<volatile u32*>(params.stop) == 0) {
            device_ring_relax(128);
          }
        }
        ++remote_reads;
        ready = params.fetch_status[status_index] > 0 &&
          valid_graph_record(params, destination);
        if (ready) break;
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
                                     f32* beam_distances, u8* beam_expanded,
                                     u32& beam_count) {
  const u32 count = min(load_cg(params.delta_count), params.delta_capacity);
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
  candidate_distances[threadIdx.x] = local_approximation;
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
        beam_insert(beam_handles, beam_distances, beam_expanded, beam_count,
                    params.beam_width, handle, candidate_distances[index]);
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
  f32* beam_distances = params.beam_distances +
    static_cast<size_t>(query_slot) * params.beam_width;
  u8* beam_expanded = params.beam_expanded +
    static_cast<size_t>(query_slot) * params.beam_width;
  u32* visited = params.visited_hash +
    static_cast<size_t>(query_slot) * params.visited_capacity;
  for (u32 index = threadIdx.x; index < params.beam_width; index += blockDim.x) {
    beam_handles[index] = UINT32_MAX;
    beam_distances[index] = FLT_MAX;
    beam_expanded[index] = 0;
  }
  for (u32 index = threadIdx.x; index < params.visited_capacity; index += blockDim.x) {
    visited[index] = UINT32_MAX;
  }
  __syncthreads();

  __shared__ u32 beam_count;
  __shared__ f32 entry_distances[kPersistentMaxEntryPoints];
  for (u32 index = threadIdx.x; index < params.entry_point_count; index += blockDim.x) {
    entry_distances[index] = approximate_handle(
      params, query_lut, query_norm2, params.entry_points[index], descriptor.snapshot_epoch);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    beam_count = 0;
    for (u32 index = 0; index < params.entry_point_count; ++index) {
      beam_insert(beam_handles, beam_distances, beam_expanded, beam_count,
                  params.beam_width, params.entry_points[index], entry_distances[index]);
    }
    if (beam_count == 0) {
      beam_insert(beam_handles, beam_distances, beam_expanded, beam_count,
                  params.beam_width, params.medoid_ordinal,
                  approximate_handle(params, query_lut, query_norm2,
                                     params.medoid_ordinal, descriptor.snapshot_epoch));
    }
    for (u32 index = 0; index < beam_count; ++index) {
      insert_visited(visited, params.visited_capacity, beam_handles[index]);
    }
  }
  __syncthreads();

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
        device_ring_push(params.completions, completion);
      }
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
        for (u32 neighbor = 0; neighbor < neighbor_counts[lane]; ++neighbor) {
          const u32 offset = lane * kPersistentMaxGraphDegree + neighbor;
          beam_insert(beam_handles, beam_distances, beam_expanded, beam_count,
                      params.beam_width, neighbor_handles[offset], neighbor_distances[offset]);
        }
      }
      expansions += selected_count;
    }
    __syncthreads();
  }

  add_delta_candidates(params, descriptor, query, query_lut, query_norm2,
                       beam_handles, beam_distances, beam_expanded, beam_count);
  __shared__ u32 exact_count;
  __shared__ u32 rerank_ids[kPersistentMaxExact];
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
        const f32 distance = beam_distances[index];
        beam_distances[index] = beam_distances[best];
        beam_distances[best] = distance;
      }
    }
    exact_count = min(beam_count, params.exact_width);
    for (u32 index = 0; index < exact_count; ++index) {
      params.fetch_status[query_slot * params.fetch_status_stride + index] = 0;
      rerank_ids[index] = UINT32_MAX;
    }
  }
  __syncthreads();

  for (u32 index = threadIdx.x; index < exact_count; index += blockDim.x) {
    const u32 handle = beam_handles[index];
    if ((handle & kDeltaHandleBit) != 0) {
      const u32 delta_slot = handle & kDeltaHandleMask;
      const DeviceDeltaRecord& record = params.delta_records[delta_slot];
      if (delta_visible(record, descriptor.snapshot_epoch)) {
        rerank_ids[index] = record.id;
        beam_distances[index] = exact_float_distance(
          params, query, params.delta_vectors + static_cast<size_t>(delta_slot) * params.dim);
      } else {
        beam_distances[index] = FLT_MAX;
      }
      continue;
    }
    u64 raw = 0;
    u64 graph_offset = 0;
    u32 shard = 0;
    if (!resolve_handle(params, handle, raw, shard, graph_offset)) {
      beam_distances[index] = FLT_MAX;
      continue;
    }
    u8* destination = params.exact_records +
      (static_cast<size_t>(query_slot) * params.exact_width + index) * params.node_record_bytes;
    const u64 node_offset = ((raw << 16) >> 16) + params.node_meta_offset;
    const u32 status_index = query_slot * params.fetch_status_stride + index;
    if (params.direct_backend != 0) {
      const i32 status = direct_fetch(params, shard, node_offset, destination,
                                      params.node_record_bytes, query_slot + index);
      params.fetch_status[status_index] = status == 0 ? 1 : status;
    } else {
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
      while (*reinterpret_cast<volatile i32*>(params.fetch_status + status_index) == 0 &&
             *reinterpret_cast<volatile u32*>(params.stop) == 0) {
        device_ring_relax(128);
      }
    }
    if (params.fetch_status[status_index] > 0) {
      rerank_ids[index] = *reinterpret_cast<const u32*>(destination);
      beam_distances[index] = exact_storage_distance(params, query, destination + 8);
    } else {
      beam_distances[index] = FLT_MAX;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (u32 index = 0; index < exact_count; ++index) {
      u32 best = index;
      for (u32 candidate = index + 1; candidate < exact_count; ++candidate) {
        if (beam_distances[candidate] < beam_distances[best]) best = candidate;
      }
      if (best != index) {
        const f32 distance = beam_distances[index];
        beam_distances[index] = beam_distances[best];
        beam_distances[best] = distance;
        const u32 id = rerank_ids[index];
        rerank_ids[index] = rerank_ids[best];
        rerank_ids[best] = id;
      }
    }
    u32 valid = 0;
    while (valid < exact_count && rerank_ids[valid] != UINT32_MAX &&
           isfinite(beam_distances[valid])) ++valid;
    const u32 result_count = min(static_cast<u32>(descriptor.k), valid);
    u32* output_ids = reinterpret_cast<u32*>(descriptor.result_device_address);
    f32* output_distances = params.result_distances +
      static_cast<size_t>(query_slot) * descriptor.result_capacity;
    for (u32 index = 0; index < result_count; ++index) {
      output_ids[index] = rerank_ids[index];
      output_distances[index] = beam_distances[index];
    }
    completion.result_count = result_count;
    completion.status = 0;
    completion.gpu_cycles = clock64() - query_started_cycles;
    completion.remote_pages = total_remote_reads;
    completion.cache_hits = total_cache_hits;
    completion.exact_vectors = exact_count;
    device_ring_push(params.completions, completion);
  }
}

__global__ void persistent_search_kernel(PersistentKernelParams params) {
  while (*reinterpret_cast<volatile u32*>(params.stop) == 0) {
    QueryDescriptor descriptor;
    if (!device_ring_try_pop(params.submissions, descriptor)) {
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

__global__ void bump_cache_generation_kernel(u64* generation) {
  if (threadIdx.x == 0) atomicAdd(reinterpret_cast<unsigned long long*>(generation), 1ULL);
}

__global__ void direct_code_bootstrap_kernel(PersistentKernelParams params,
                                             const FetchDescriptor* requests,
                                             i32* statuses,
                                             u32 count) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const FetchDescriptor& request = requests[index];
  const i32 status = direct_fetch(
    params, request.memory_node, request.remote_offset,
    reinterpret_cast<u8*>(request.destination_address), request.bytes, index);
  statuses[index] = status == 0 ? 1 : status;
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

void launch_bump_graph_cache_generation(cudaStream_t stream, u64* generation) {
  bump_cache_generation_kernel<<<1, 1, 0, stream>>>(generation);
}

void launch_direct_code_bootstrap(cudaStream_t stream,
                                  const PersistentKernelParams& params,
                                  const FetchDescriptor* requests,
                                  i32* statuses,
                                  u32 count) {
  direct_code_bootstrap_kernel<<<(count + 31) / 32, 32, 0, stream>>>(
    params, requests, statuses, count);
}

}  // namespace gpu_search
