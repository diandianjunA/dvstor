#pragma once

#include "gpu_search/persistent_kernel/query_traversal.cuh"

namespace gpu_search::persistent_kernel_detail {

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

}  // namespace gpu_search::persistent_kernel_detail
