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

__device__ __forceinline__ u32 hash_id(u32 value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

__device__ __forceinline__ u32 hash_page(u64 value) {
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdULL;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ULL;
  value ^= value >> 33;
  return static_cast<u32>(value ^ (value >> 32));
}

struct DevicePageHeader {
  u32 magic;
  u16 version;
  u16 node_count;
  u32 payload_bytes;
  u32 generation;
};

struct DevicePageNodeHeader {
  u32 node_id;
  u16 degree;
  u16 flags;
};

__device__ __forceinline__ u32 decode_neighbor_id(const u8* source, u32 bytes) {
  u32 id = static_cast<u32>(source[0]) |
    (static_cast<u32>(source[1]) << 8) |
    (static_cast<u32>(source[2]) << 16);
  if (bytes == 4) id |= static_cast<u32>(source[3]) << 24;
  return id;
}

__device__ bool insert_visited(u32* table, u32 capacity, u32 id) {
  const u32 mask = capacity - 1;
  u32 slot = hash_id(id) & mask;
  for (u32 probe = 0; probe < capacity; ++probe) {
    const u32 old = atomicCAS(&table[slot], UINT32_MAX, id);
    if (old == UINT32_MAX) return true;
    if (old == id) return false;
    slot = (slot + 1) & mask;
  }
  return false;
}

__device__ f32 approximate_entry(const PersistentKernelParams& params,
                                 const f32* rotated_query,
                                 f32 query_norm2,
                                 const u8* entry) {
  f32 signed_dot = 0.0f;
  for (u32 bit = 0; bit < params.code_bits; ++bit) {
    const bool positive = (entry[bit >> 3] & static_cast<u8>(1u << (7u - (bit & 7u)))) != 0;
    signed_dot += positive ? rotated_query[bit] : -rotated_query[bit];
  }
  f32 norm = 0.0f;
  f32 error = 1.0f;
  memcpy(&norm, entry + params.code_storage_bytes, sizeof(norm));
  memcpy(&error, entry + params.code_storage_bytes + sizeof(norm), sizeof(error));
  const f32 denominator = sqrtf(static_cast<f32>(params.code_bits)) * fmaxf(error, 1e-6f);
  const f32 inner_product = norm * signed_dot / denominator;
  return fmaxf(query_norm2 + norm * norm - 2.0f * inner_product, 0.0f);
}

__device__ f32 approximate_distance(const PersistentKernelParams& params,
                                    const f32* rotated_query,
                                    f32 query_norm2,
                                    u32 id) {
  if (id >= params.num_nodes || (params.nodes[id].flags & kDeltaDeleted) != 0) return FLT_MAX;
  const u8* entry = params.rabitq_entries +
    static_cast<size_t>(id) * params.rabitq_entry_bytes;
  return approximate_entry(params, rotated_query, query_norm2, entry);
}

__device__ void beam_insert(u32* ids, f32* distances, u8* expanded,
                            u32& count, u32 capacity, u32 id, f32 distance) {
  if (!isfinite(distance) || distance == FLT_MAX) return;
  if (count < capacity) {
    ids[count] = id;
    distances[count] = distance;
    expanded[count] = 0;
    ++count;
    return;
  }
  u32 worst = 0;
  for (u32 i = 1; i < count; ++i) {
    if (distances[i] > distances[worst]) worst = i;
  }
  if (distance >= distances[worst]) return;
  ids[worst] = id;
  distances[worst] = distance;
  expanded[worst] = 0;
}

__device__ f32 exact_distance(const PersistentKernelParams& params,
                              const f32* query, const u8* vector) {
  f32 distance = 0.0f;
  for (u32 d = 0; d < params.dim; ++d) {
    f32 component = 0.0f;
    if (params.vector_dtype == 0) {
      component = reinterpret_cast<const f32*>(vector)[d];
    } else if (params.vector_dtype == 1) {
      component = static_cast<f32>(vector[d]);
    } else {
      component = static_cast<f32>(reinterpret_cast<const int8_t*>(vector)[d]);
    }
    const f32 difference = query[d] - component;
    distance += difference * difference;
  }
  return distance;
}

__device__ f32 exact_float_distance(const PersistentKernelParams& params,
                                    const f32* query, const f32* vector) {
  f32 distance = 0.0f;
  for (u32 d = 0; d < params.dim; ++d) {
    const f32 difference = query[d] - vector[d];
    distance += difference * difference;
  }
  return distance;
}

__device__ i32 direct_fetch(const PersistentKernelParams& params,
                            u32 memory_node, u64 remote_offset,
                            u8* destination, u32 bytes, u32 lane) {
#ifdef DVSTOR_HAVE_GPUNETIO
  if (memory_node >= params.direct_region_count || params.direct_qps == nullptr ||
      params.direct_qps_per_node == 0) {
    return -EINVAL;
  }
  const u32 qp_index = (lane % params.direct_qps_per_node) *
    params.direct_region_count + memory_node;
  if (params.direct_qps[qp_index] == nullptr) return -EINVAL;
  auto* qp = reinterpret_cast<doca_gpu_dev_verbs_qp*>(params.direct_qps[qp_index]);
  const DirectRemoteRegion& region = params.direct_regions[memory_node];
  doca_gpu_dev_verbs_addr remote{
    .addr = region.address + remote_offset,
    .key = region.rkey,
  };
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

__device__ void process_query(const PersistentKernelParams& params,
                              const QueryDescriptor& query_descriptor) {
  const u32 slot = query_descriptor.query_slot;
  __shared__ u64 query_started_cycles;
  if (threadIdx.x == 0) query_started_cycles = clock64();
  __syncthreads();
  CompletionDescriptor completion{
    .request_id = query_descriptor.request_id,
    .snapshot_epoch = query_descriptor.snapshot_epoch,
    .query_slot = slot,
  };
  if (slot >= params.query_slots || query_descriptor.dim != params.dim ||
      query_descriptor.k == 0 || query_descriptor.k > query_descriptor.result_capacity) {
    if (threadIdx.x == 0) {
      completion.status = -EINVAL;
      completion.gpu_cycles = clock64() - query_started_cycles;
      device_ring_push(params.completions, completion);
    }
    return;
  }

  const f32* query = reinterpret_cast<const f32*>(query_descriptor.query_device_address);
  f32* rotated = params.rotated_queries + static_cast<size_t>(slot) * params.code_bits;
  __shared__ f32 query_norm2;
  if (threadIdx.x == 0) query_norm2 = 0.0f;
  __syncthreads();
  f32 local_norm2 = 0.0f;
  for (u32 d = threadIdx.x; d < params.code_bits; d += blockDim.x) {
    f32 value = d < params.dim ? query[d] - params.centroid[d] : 0.0f;
    if (d < params.dim) {
      local_norm2 += value * value;
      u32 hash = d + 0x9e3779b9U;
      hash ^= hash >> 16;
      hash *= 0x7feb352dU;
      hash ^= hash >> 15;
      value = (hash & 1U) ? value : -value;
    }
    rotated[d] = value;
  }
  atomicAdd(&query_norm2, local_norm2);
  __syncthreads();
  for (u32 width = 1; width < params.code_bits; width <<= 1) {
    const u32 butterflies = params.code_bits >> 1;
    for (u32 index = threadIdx.x; index < butterflies; index += blockDim.x) {
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
  const f32 scale = rsqrtf(static_cast<f32>(params.code_bits));
  for (u32 d = threadIdx.x; d < params.code_bits; d += blockDim.x) rotated[d] *= scale;
  __shared__ u32 query_signature;
  if (threadIdx.x == 0) {
    query_signature = 0;
    for (u32 bit = 0; bit < 16; ++bit) {
      const u32 source = bit * params.code_bits / 16;
      if (rotated[source] > 0.0f) query_signature |= 1u << bit;
    }
  }
  __syncthreads();

  u32* beam_ids = params.beam_ids + static_cast<size_t>(slot) * params.beam_width;
  f32* beam_distances = params.beam_distances + static_cast<size_t>(slot) * params.beam_width;
  u8* beam_expanded = params.beam_expanded + static_cast<size_t>(slot) * params.beam_width;
  u32* visited = params.visited_hash + static_cast<size_t>(slot) * params.visited_capacity;
  for (u32 i = threadIdx.x; i < params.beam_width; i += blockDim.x) {
    beam_ids[i] = UINT32_MAX;
    beam_distances[i] = FLT_MAX;
    beam_expanded[i] = 0;
  }
  for (u32 i = threadIdx.x; i < params.visited_capacity; i += blockDim.x) {
    visited[i] = UINT32_MAX;
  }
  __syncthreads();

  __shared__ u32 beam_count;
  __shared__ i32 best_index;
  __shared__ u32 neighbor_ids[256];
  __shared__ f32 neighbor_distances[256];
  __shared__ u32 neighbor_count;
  __shared__ u32 expanded_node_id;
  __shared__ u32 remote_pages;
  __shared__ u32 cache_hits;
  __shared__ f32 entry_distances[kPersistentMaxEntryPoints];
  for (u32 i = threadIdx.x; i < params.entry_point_count; i += blockDim.x) {
    entry_distances[i] = approximate_distance(
      params, rotated, query_norm2, params.entry_points[i]);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    beam_count = 0;
    remote_pages = 0;
    cache_hits = 0;
    for (u32 i = 0; i < params.entry_point_count; ++i) {
      beam_insert(beam_ids, beam_distances, beam_expanded, beam_count,
                  params.beam_width, params.entry_points[i], entry_distances[i]);
    }
    if (beam_count == 0) {
      beam_insert(beam_ids, beam_distances, beam_expanded, beam_count,
                  params.beam_width, params.medoid_id,
                  approximate_distance(params, rotated, query_norm2, params.medoid_id));
    }
    for (u32 i = 0; i < beam_count; ++i) {
      insert_visited(visited, params.visited_capacity, beam_ids[i]);
    }
  }
  __syncthreads();

  for (u32 expansion = 0; expansion < params.max_expansions; ++expansion) {
    if (threadIdx.x == 0) {
      best_index = -1;
      f32 best_distance = FLT_MAX;
      for (u32 i = 0; i < beam_count; ++i) {
        if (!beam_expanded[i] && beam_distances[i] < best_distance) {
          best_distance = beam_distances[i];
          best_index = static_cast<i32>(i);
        }
      }
      if (best_index >= 0) {
        beam_expanded[best_index] = 1;
        const u32 id = beam_ids[best_index];
        expanded_node_id = id;
        const auto& node = params.nodes[id];
        neighbor_count = min(static_cast<u32>(node.hot_neighbor_count), kPersistentMaxHotDegree);
        for (u32 i = 0; i < neighbor_count; ++i) {
          neighbor_ids[i] = params.hot_neighbors[node.hot_neighbor_begin + i];
        }
      } else {
        neighbor_count = 0;
      }
    }
    __syncthreads();
    if (best_index < 0) break;

    if (threadIdx.x == 0 && expansion < params.cold_expansions &&
        params.graph_page_cache_slots != 0 &&
        params.graph_page_cache != nullptr &&
        params.graph_page_cache_keys != nullptr &&
        params.graph_page_cache_locks != nullptr) {
      const DeviceNodeRecord& node = params.nodes[expanded_node_id];
      if (node.cold_page_offset != 0 && node.cold_record_offset != 0) {
        const u64 key = (static_cast<u64>(node.shard) << 56) ^ node.cold_page_offset;
        const u32 cache_slot = hash_page(key) % params.graph_page_cache_slots;
        bool lock_acquired = false;
        while (!lock_acquired && *reinterpret_cast<volatile u32*>(params.stop) == 0) {
          lock_acquired = atomicCAS(params.graph_page_cache_locks + cache_slot, 0u, 1u) == 0;
          if (!lock_acquired) device_ring_relax(128);
        }
        if (lock_acquired) {
          u8* page = params.graph_page_cache +
            static_cast<size_t>(cache_slot) * params.graph_page_bytes;
          bool page_ready = params.graph_page_cache_keys[cache_slot] == key;
          if (page_ready) {
            ++cache_hits;
          } else {
            const u32 status_index = slot * params.fetch_status_stride + params.exact_width;
            params.fetch_status[status_index] = 0;
            if (params.direct_backend) {
              const i32 status = direct_fetch(params, node.shard, node.cold_page_offset,
                                                page, params.graph_page_bytes,
                                                slot + expansion);
              params.fetch_status[status_index] = status == 0 ? 1 : status;
            } else {
              FetchDescriptor fetch{
                .request_id = query_descriptor.request_id,
                .remote_offset = node.cold_page_offset,
                .destination_address = reinterpret_cast<u64>(page),
                .bytes = params.graph_page_bytes,
                .memory_node = node.shard,
                .kind = static_cast<u8>(FetchKind::graph_page),
                .sequence = status_index,
              };
              device_ring_push(params.fetches, fetch);
              while (*reinterpret_cast<volatile i32*>(&params.fetch_status[status_index]) == 0) {
                if (*reinterpret_cast<volatile u32*>(params.stop) != 0) break;
                device_ring_relax(128);
              }
            }
            ++remote_pages;
            page_ready = params.fetch_status[status_index] > 0;
            if (page_ready) {
              __threadfence();
              atomicExch(reinterpret_cast<unsigned long long*>(
                           params.graph_page_cache_keys + cache_slot), key);
            }
          }
          if (page_ready && node.cold_record_offset + sizeof(DevicePageNodeHeader) <=
                                params.graph_page_bytes) {
            const auto* page_header = reinterpret_cast<const DevicePageHeader*>(page);
            const auto* node_header = reinterpret_cast<const DevicePageNodeHeader*>(
              page + node.cold_record_offset);
            const u32 degree = min(static_cast<u32>(node_header->degree), 255u);
            const u64 record_end = static_cast<u64>(node.cold_record_offset) +
              sizeof(DevicePageNodeHeader) +
              static_cast<u64>(degree) * params.id_encoding_bytes;
            if (page_header->magic == 0x47504750u && page_header->version == 1 &&
                node_header->node_id == expanded_node_id &&
                (params.id_encoding_bytes == 3 || params.id_encoding_bytes == 4) &&
                record_end <= params.graph_page_bytes) {
              const u8* encoded = reinterpret_cast<const u8*>(node_header + 1);
              neighbor_count = degree;
              for (u32 i = 0; i < degree; ++i) {
                neighbor_ids[i] = decode_neighbor_id(
                  encoded + static_cast<size_t>(i) * params.id_encoding_bytes,
                  params.id_encoding_bytes);
              }
            }
          }
          __threadfence();
          atomicExch(params.graph_page_cache_locks + cache_slot, 0u);
        }
      }
    }
    __syncthreads();

    for (u32 i = threadIdx.x; i < neighbor_count; i += blockDim.x) {
      const u32 id = neighbor_ids[i];
      neighbor_distances[i] = id < params.num_nodes &&
          insert_visited(visited, params.visited_capacity, id)
        ? approximate_distance(params, rotated, query_norm2, id) : FLT_MAX;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (u32 i = 0; i < neighbor_count; ++i) {
        beam_insert(beam_ids, beam_distances, beam_expanded, beam_count,
                    params.beam_width, neighbor_ids[i], neighbor_distances[i]);
      }
    }
    __syncthreads();
  }

  __shared__ u32 exact_count;
  __shared__ u32 remote_exact_count;
  if (threadIdx.x == 0) {
    for (u32 i = 0; i < beam_count; ++i) {
      u32 best = i;
      for (u32 j = i + 1; j < beam_count; ++j) {
        if (beam_distances[j] < beam_distances[best]) best = j;
      }
      if (best != i) {
        const u32 id = beam_ids[i];
        beam_ids[i] = beam_ids[best];
        beam_ids[best] = id;
        const f32 distance = beam_distances[i];
        beam_distances[i] = beam_distances[best];
        beam_distances[best] = distance;
      }
    }
    exact_count = min(min(beam_count, params.exact_width),
                      static_cast<u32>(query_descriptor.result_capacity));
    remote_exact_count = exact_count;
    for (u32 i = 0; i < exact_count; ++i) {
      const u32 status_index = slot * params.fetch_status_stride + i;
      params.fetch_status[status_index] = 0;
    }
    if (!params.direct_backend) {
      u8* vector_base = params.exact_vectors +
        static_cast<size_t>(slot) * params.exact_width * params.vector_bytes;
      for (u32 i = 0; i < exact_count; ++i) {
        const u32 status_index = slot * params.fetch_status_stride + i;
      const DeviceNodeRecord& node = params.nodes[beam_ids[i]];
      const u64 raw = node.remote_node;
      FetchDescriptor fetch{
        .request_id = query_descriptor.request_id,
        .remote_offset = (raw << 16 >> 16) + params.vector_offset,
        .destination_address = reinterpret_cast<u64>(vector_base +
          static_cast<size_t>(i) * params.vector_bytes),
        .bytes = params.vector_bytes,
        .memory_node = static_cast<u16>(raw >> 48),
        .kind = static_cast<u8>(FetchKind::vector),
        .sequence = status_index,
      };
      device_ring_push(params.fetches, fetch);
      }
    }
  }
  __syncthreads();

  for (u32 i = threadIdx.x; i < exact_count; i += blockDim.x) {
    const u32 status_index = slot * params.fetch_status_stride + i;
    if (params.direct_backend) {
      const DeviceNodeRecord& node = params.nodes[beam_ids[i]];
      const u64 raw = node.remote_node;
      u8* destination = params.exact_vectors +
        (static_cast<size_t>(slot) * params.exact_width + i) * params.vector_bytes;
      const i32 status = direct_fetch(params, static_cast<u32>(raw >> 48),
                                      (raw << 16 >> 16) + params.vector_offset,
                                      destination, params.vector_bytes, slot + i);
      params.fetch_status[status_index] = status == 0 ? 1 : status;
    } else {
      while (*reinterpret_cast<volatile i32*>(&params.fetch_status[status_index]) == 0) {
        if (*reinterpret_cast<volatile u32*>(params.stop) != 0) break;
        device_ring_relax(128);
      }
    }
    if (params.fetch_status[status_index] > 0) {
      const u8* vector = params.exact_vectors +
        (static_cast<size_t>(slot) * params.exact_width + i) * params.vector_bytes;
      beam_distances[i] = exact_distance(params, query, vector);
    } else {
      beam_distances[i] = FLT_MAX;
    }
  }
  __syncthreads();

  __shared__ u32 delta_candidate_ids[128];
  __shared__ f32 delta_candidate_distances[128];
  u32 best_delta_slot = UINT32_MAX;
  f32 best_delta_approximation = FLT_MAX;
  if (params.delta_count != nullptr && params.delta_records != nullptr) {
    const u32 count = min(*params.delta_count, params.delta_capacity);
    for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
      const DeviceDeltaRecord& record = params.delta_records[index];
      const bool visible = record.epoch <= query_descriptor.snapshot_epoch &&
        (record.superseded_epoch == 0 ||
         record.superseded_epoch > query_descriptor.snapshot_epoch);
      if (!visible || (record.flags & kDeltaDeleted) != 0) continue;
      if (count > 65536 && __popc((record.signature ^ query_signature) & 0xffffu) > 4) {
        continue;
      }
      const u8* entry = params.delta_rabitq_entries +
        static_cast<size_t>(index) * params.rabitq_entry_bytes;
      const f32 approximation = approximate_entry(params, rotated, query_norm2, entry);
      if (approximation < best_delta_approximation) {
        best_delta_approximation = approximation;
        best_delta_slot = index;
      }
    }
  }
  if (threadIdx.x < 128) {
    if (best_delta_slot != UINT32_MAX) {
      const DeviceDeltaRecord& record = params.delta_records[best_delta_slot];
      delta_candidate_ids[threadIdx.x] = record.id;
      delta_candidate_distances[threadIdx.x] = exact_float_distance(
        params, query, params.delta_vectors +
          static_cast<size_t>(best_delta_slot) * params.dim);
    } else {
      delta_candidate_ids[threadIdx.x] = UINT32_MAX;
      delta_candidate_distances[threadIdx.x] = FLT_MAX;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (u32 i = 0; i < exact_count; ++i) {
      const u32 id = beam_ids[i];
      if (id < params.num_nodes && params.base_override_epochs != nullptr) {
        const u64 override_epoch = params.base_override_epochs[id];
        if (override_epoch != 0 && override_epoch <= query_descriptor.snapshot_epoch) {
          beam_distances[i] = FLT_MAX;
        }
      }
    }
    u32 combined_count = exact_count;
    for (u32 candidate = 0; candidate < min(blockDim.x, 128u); ++candidate) {
      const u32 id = delta_candidate_ids[candidate];
      const f32 distance = delta_candidate_distances[candidate];
      if (id == UINT32_MAX || !isfinite(distance)) continue;
      bool duplicate = false;
      for (u32 i = 0; i < combined_count; ++i) {
        if (beam_ids[i] != id) continue;
        beam_distances[i] = fminf(beam_distances[i], distance);
        duplicate = true;
        break;
      }
      if (!duplicate && combined_count < params.beam_width) {
        beam_ids[combined_count] = id;
        beam_distances[combined_count] = distance;
        ++combined_count;
      }
    }
    exact_count = combined_count;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (u32 i = 0; i < exact_count; ++i) {
      u32 best = i;
      for (u32 j = i + 1; j < exact_count; ++j) {
        if (beam_distances[j] < beam_distances[best]) best = j;
      }
      if (best != i) {
        const u32 id = beam_ids[i];
        beam_ids[i] = beam_ids[best];
        beam_ids[best] = id;
        const f32 distance = beam_distances[i];
        beam_distances[i] = beam_distances[best];
        beam_distances[best] = distance;
      }
    }
    u32 valid_count = 0;
    while (valid_count < exact_count && beam_ids[valid_count] != UINT32_MAX &&
           isfinite(beam_distances[valid_count])) {
      ++valid_count;
    }
    const u32 result_count = min(static_cast<u32>(query_descriptor.k), valid_count);
    u32* output_ids = reinterpret_cast<u32*>(query_descriptor.result_device_address);
    f32* output_distances = params.result_distances +
      static_cast<size_t>(slot) * query_descriptor.result_capacity;
    for (u32 i = 0; i < result_count; ++i) {
      output_ids[i] = beam_ids[i];
      output_distances[i] = beam_distances[i];
    }
    completion.result_count = result_count;
    completion.status = 0;
    completion.gpu_cycles = clock64() - query_started_cycles;
    completion.remote_pages = remote_pages;
    completion.cache_hits = cache_hits;
    completion.exact_vectors = remote_exact_count;
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

}  // namespace

void launch_persistent_search(cudaStream_t stream, const PersistentKernelParams& params,
                              u32 blocks, u32 threads) {
  persistent_search_kernel<<<blocks, threads, 0, stream>>>(params);
}

}  // namespace gpu_search
