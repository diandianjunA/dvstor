#pragma once

#include "gpu_search/persistent_kernel/query_traversal.cuh"

namespace gpu_search::persistent_kernel_detail {

template <bool EnableAsfe>
__device__ void direct_read_owner_loop(const PersistentKernelParams& params,
                                       u32 queue_count, u32 owner_block);

template <u32 Threads, bool EnableAsfe,
          u32 PqSubquantizers = kPersistentRuntimePqSubquantizers>
__global__ __launch_bounds__(
  Threads,
  Threads == 128
    ? 3
    : 1) void persistent_search_kernel(const __grid_constant__
                                         PersistentKernelParams params) {
  static_assert(Threads == 128 || Threads == 256);
  const bool unified_dispatch = params.direct_owner_block_count != 0;
  if (unified_dispatch && blockIdx.x < params.direct_owner_block_count) {
    direct_read_owner_loop<EnableAsfe>(params, params.direct_batch_queue_count,
                                       blockIdx.x);
    return;
  }

  constexpr u32 kQueryRole = 1u << 0;
  constexpr u32 kDispatcherRole = 1u << 1;
  constexpr u32 kRouteControlRole = 1u << 2;
  // The CTA role is uniform and immutable for the lifetime of this
  // persistent kernel. A single bitmask avoids keeping three independent
  // booleans live across the permanent loop. In contrast to a shared role
  // word, ptxas keeps this one scalar in-register without spilling, so the
  // polling loop does not trade the old local loads for repeated LDS.
  const u32 role_block = blockIdx.x - params.direct_owner_block_count;
  const u32 block_roles =
    !unified_dispatch                        ? kQueryRole | kRouteControlRole
    : role_block < params.query_block_count  ? kQueryRole
    : role_block == params.query_block_count ? kDispatcherRole
    : role_block == params.query_block_count + 1 ? kRouteControlRole
                                                 : 0u;
  if (block_roles == 0) return;

  if (threadIdx.x == 0) {
    if (unified_dispatch) {
      u32* ready_count = (block_roles & kQueryRole) != 0
                           ? params.query_kernel_ready_count
                         : (block_roles & kDispatcherRole) != 0
                           ? params.dispatcher_kernel_ready_count
                           : params.control_kernel_ready_count;
      if (ready_count != nullptr) atomicAdd(ready_count, 1u);
      __threadfence_system();
    } else if (params.kernel_ready_count != nullptr) {
      atomicAdd(params.kernel_ready_count, 1u);
      __threadfence_system();
    }
  }

  __shared__ QueryDescriptor descriptor;
  __shared__ QueryDescriptor dispatch_descriptor;
  __shared__ CentroidRoutePublishDescriptor route_descriptor;
  __shared__ u32 have_submission;
  __shared__ u32 dispatch_pending;
  __shared__ u32 have_route_submission;
  __shared__ u32 stop_requested;
  __shared__ u32 idle_cycles;
  __shared__ i32 route_status;
  // One controller belongs to one persistent query CTA, not one query.  Its
  // learned tail width and collapsed-query re-probe cadence therefore survive
  // successive submissions without any global state or atomic operation.
  __shared__ adaptive_frontier::ControllerState frontier_controller;
  if (threadIdx.x == 0) {
    dispatch_pending = 0;
    idle_cycles = 256u + ((blockIdx.x * 131u) & 1023u);
    frontier_controller = adaptive_frontier::make_controller_state(
      params.commit_width, params.issue_width);
  }
  __syncthreads();

  for (;;) {
    if (threadIdx.x == 0) {
      stop_requested = *reinterpret_cast<volatile u32*>(params.stop);
    }
    __syncthreads();
    if (stop_requested != 0) return;

    if ((block_roles & kDispatcherRole) != 0) {
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
      have_route_submission =
        (block_roles & kRouteControlRole) != 0 &&
            params.route_submissions.entries != nullptr &&
            device_ring_try_pop(params.route_submissions, route_descriptor)
          ? 1u
          : 0u;
    }
    __syncthreads();
    if (have_route_submission != 0) {
      if (threadIdx.x == 0) {
        route_status = 0;
        if (route_descriptor.command_id == 0 ||
            route_descriptor.update_count == 0 ||
            route_descriptor.update_count >
              params.centroid_route_shard_capacity ||
            params.centroid_route_updates == nullptr ||
            params.centroid_route_centroid_updates == nullptr ||
            params.centroid_route_shards == nullptr ||
            params.centroid_route_entries == nullptr ||
            params.shard_centroids == nullptr ||
            params.centroid_route_epoch == nullptr ||
            params.centroid_route_shard_capacity == 0 ||
            params.centroid_route_entry_capacity == 0 ||
            params.centroid_route_entry_capacity >
              kCentroidRouteMaxLiveEntries) {
          route_status = -EINVAL;
        }
      }
      __syncthreads();

      if (route_status == 0) {
        for (u32 index = threadIdx.x; index < route_descriptor.update_count;
             index += blockDim.x) {
          const CentroidRouteUpdate update =
            params.centroid_route_updates[index];
          bool duplicate_shard = false;
          for (u32 prior = 0; prior < index; ++prior) {
            duplicate_shard =
              duplicate_shard ||
              params.centroid_route_updates[prior].shard == update.shard;
          }
          bool invalid_entry = false;
          for (u32 entry = 0; entry < update.live_entry_count; ++entry) {
            const DeviceCentroidRouteEntry& candidate = update.entries[entry];
            invalid_entry = invalid_entry || candidate.remote_node == 0 ||
                            candidate.flags != kCentroidRouteLive ||
                            remote_shard(candidate.remote_node) != update.shard;
            for (u32 prior = 0; prior < entry; ++prior) {
              invalid_entry =
                invalid_entry ||
                update.entries[prior].remote_node == candidate.remote_node;
            }
          }
          if (update.shard >= params.num_shards ||
              update.shard >= params.centroid_route_shard_capacity ||
              update.version == 0 ||
              update.live_entry_count > params.centroid_route_entry_capacity ||
              ((update.vector_count == 0) != (update.live_entry_count == 0)) ||
              duplicate_shard || invalid_entry) {
            atomicExch(&route_status, -EINVAL);
            continue;
          }
          const DeviceCentroidRouteShard& current =
            params.centroid_route_shards[update.shard];
          const u64 current_command =
            centroid_route_atomic_load(current.command_id);
          const u64 current_version =
            centroid_route_atomic_load(current.version);
          if (current_command >= route_descriptor.command_id ||
              current_version > update.version) {
            atomicExch(&route_status, -ESTALE);
          }
        }
      }
      __syncthreads();

      if (route_status == 0) {
        if (threadIdx.x == 0) {
          cuda::atomic_ref<u64, cuda::thread_scope_device> route_epoch(
            *params.centroid_route_epoch);
          const u64 previous =
            route_epoch.fetch_add(1, cuda::memory_order_acq_rel);
          // Only the dedicated control CTA writes this epoch, so observing an
          // odd value indicates memory corruption rather than contention.
          if ((previous & 1u) != 0) {
            atomicExch(&route_status, -EIO);
          }
        }
        __syncthreads();
      }

      if (route_status == 0) {
        // One shard seqlock covers its centroid and complete live-entry set.
        for (u32 index = threadIdx.x; index < route_descriptor.update_count;
             index += blockDim.x) {
          const CentroidRouteUpdate update =
            params.centroid_route_updates[index];
          DeviceCentroidRouteShard& destination =
            params.centroid_route_shards[update.shard];
          cuda::atomic_ref<u64, cuda::thread_scope_device> sequence(
            destination.sequence);
          sequence.fetch_add(1, cuda::memory_order_acq_rel);
        }
        __syncthreads();

        for (u64 item = threadIdx.x;
             item <
             static_cast<u64>(route_descriptor.update_count) * params.dim;
             item += blockDim.x) {
          const u32 update_index = static_cast<u32>(item / params.dim);
          const u32 dimension = static_cast<u32>(item % params.dim);
          const CentroidRouteUpdate update =
            params.centroid_route_updates[update_index];
          params
            .shard_centroids[static_cast<size_t>(update.shard) * params.dim +
                             dimension] =
            params.centroid_route_centroid_updates[item];
        }
        for (u64 item = threadIdx.x;
             item < static_cast<u64>(route_descriptor.update_count) *
                      params.centroid_route_entry_capacity;
             item += blockDim.x) {
          const u32 update_index =
            static_cast<u32>(item / params.centroid_route_entry_capacity);
          const u32 entry =
            static_cast<u32>(item % params.centroid_route_entry_capacity);
          const CentroidRouteUpdate update =
            params.centroid_route_updates[update_index];
          params.centroid_route_entries[static_cast<size_t>(update.shard) *
                                          params.centroid_route_entry_capacity +
                                        entry] = entry < update.live_entry_count
                                                   ? update.entries[entry]
                                                   : DeviceCentroidRouteEntry{};
        }
        __threadfence();
        __syncthreads();

        for (u32 index = threadIdx.x; index < route_descriptor.update_count;
             index += blockDim.x) {
          const CentroidRouteUpdate update =
            params.centroid_route_updates[index];
          DeviceCentroidRouteShard& destination =
            params.centroid_route_shards[update.shard];
          cuda::atomic_ref<u64, cuda::thread_scope_device> sequence(
            destination.sequence);
          centroid_route_atomic_store(destination.command_id,
                                      route_descriptor.command_id);
          centroid_route_atomic_store(destination.version, update.version);
          centroid_route_atomic_store(destination.vector_count,
                                      update.vector_count);
          centroid_route_atomic_store(destination.live_entry_count,
                                      update.live_entry_count);
          __threadfence();
          sequence.fetch_add(1, cuda::memory_order_release);
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
          cuda::atomic_ref<u64, cuda::thread_scope_device> route_epoch(
            *params.centroid_route_epoch);
          route_epoch.fetch_add(1, cuda::memory_order_release);
        }
      } else if (params.centroid_route_epoch != nullptr && threadIdx.x == 0) {
        // Restore an epoch opened above if a defensive consistency check ever
        // fails after validation. No partially written shard is published by
        // that path, but leaving the epoch odd would stop all future queries.
        cuda::atomic_ref<u64, cuda::thread_scope_device> route_epoch(
          *params.centroid_route_epoch);
        const u64 current = route_epoch.load(cuda::memory_order_relaxed);
        if ((current & 1u) != 0) {
          route_epoch.fetch_add(1, cuda::memory_order_release);
        }
      }
      __syncthreads();

      if (threadIdx.x == 0) {
        device_ring_push(
          params.route_completions,
          CentroidRoutePublishCompletion{
            .command_id = route_descriptor.command_id,
            .status = route_status,
            .update_count =
              route_status == 0 ? route_descriptor.update_count : 0u,
          });
        idle_cycles = 256u;
      }
      __syncthreads();
      continue;
    }

    if (threadIdx.x == 0) {
      const DeviceRingView<QueryDescriptor> query_queue =
        params.device_submissions.entries != nullptr ? params.device_submissions
                                                     : params.submissions;
      have_submission = (block_roles & kQueryRole) != 0 &&
                            query_queue.entries != nullptr &&
                            device_ring_try_pop(query_queue, descriptor)
                          ? 1u
                          : 0u;
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
    process_query<EnableAsfe, PqSubquantizers>(
      params, descriptor, frontier_controller);
    __syncthreads();
  }
}

__device__ void complete_direct_batch(const DirectBatchDescriptor& descriptor,
                                      i32 status,
                                      DirectOwnerProgress* owner_progress,
                                      u64* submission_completion_ns = nullptr) {
  if (descriptor.completion_status != nullptr) {
    if (descriptor.completion_timestamp_ns != nullptr) {
      u64 completion_ns = 0;
      if (submission_completion_ns != nullptr) {
        if (*submission_completion_ns == 0) {
          *submission_completion_ns = global_time_ns();
        }
        completion_ns = *submission_completion_ns;
      } else {
        completion_ns = global_time_ns();
      }
      *descriptor.completion_timestamp_ns = completion_ns;
    }
    __threadfence_system();
    atomicExch(descriptor.completion_status, status);
  }
  // A malformed/future caller may have announced a descriptor without a
  // completion word. The owner has nevertheless dequeued and terminally
  // classified it, so balance the watchdog even when there is no status
  // address to publish. Normal producers always provide the word.
  if (owner_progress != nullptr) {
    record_owner_watchdog_counter(&owner_progress->completed);
  }
}

__device__ __forceinline__ DirectBatchDescriptor
make_speculative_tail_descriptor(const DirectBatchDescriptor& descriptor) {
  const u32 split = descriptor.critical_request_count;
  DirectBatchDescriptor tail = descriptor;
  tail.request_shards += split;
  tail.remote_offsets += split;
  tail.local_iova_offsets += split;
  if (tail.request_bytes != nullptr) tail.request_bytes += split;
  tail.completion_status = descriptor.speculative_completion_status;
  tail.completion_timestamp_ns = descriptor.speculative_completion_timestamp_ns;
  tail.speculative_completion_status = nullptr;
  tail.speculative_completion_timestamp_ns = nullptr;
  tail.request_count =
    static_cast<u16>(static_cast<u32>(descriptor.request_count) - split);
  tail.priority = static_cast<u8>(DirectBatchPriority::speculative);
  tail.critical_request_count = 0;
  tail.flags = 0;
  return tail;
}

__device__ __forceinline__ void complete_unadmitted_speculative_tail(
  const DirectBatchDescriptor& descriptor, i32 status) {
  if (descriptor.speculative_completion_status == nullptr) return;
  // A slack rejection never became an independently announced owner
  // operation. Publish only the query-owned tail completion word: touching
  // owner progress here would make completed exceed announced and arm the
  // watchdog permanently.
  const DirectBatchDescriptor completion_only{
    .completion_status = descriptor.speculative_completion_status,
    .completion_timestamp_ns = descriptor.speculative_completion_timestamp_ns,
  };
  complete_direct_batch(completion_only, status, nullptr);
}

__device__ __forceinline__ void complete_direct_batch_with_unadmitted_tail(
  const DirectBatchDescriptor& descriptor, i32 status,
  DirectOwnerProgress* owner_progress) {
  complete_direct_batch(descriptor, status, owner_progress);
  complete_unadmitted_speculative_tail(descriptor, status);
}

__device__ __forceinline__ bool is_mandatory_fenced_tail(
  const DirectBatchDescriptor& descriptor) {
  return (descriptor.flags & (kDirectBatchFlagMandatoryFencedTail |
                              kDirectBatchFlagMixedMandatoryFencedTail)) != 0;
}

__device__ __forceinline__ bool is_mixed_mandatory_fenced_tail(
  const DirectBatchDescriptor& descriptor) {
  return (descriptor.flags & kDirectBatchFlagMixedMandatoryFencedTail) != 0;
}

// Mandatory descriptors deliberately avoid a per-request byte array: every
// prefix entry is one full exact record and every suffix entry is one u64
// header. This is also essential because the owner CTA cannot dereference a
// query CTA's shared-memory scratch.
__device__ __forceinline__ u32 direct_batch_request_length(
  const DirectBatchDescriptor& descriptor, u32 index) {
  if (is_mandatory_fenced_tail(descriptor) &&
      index >= descriptor.critical_request_count) {
    return sizeof(u64);
  }
  return descriptor.request_bytes == nullptr ? descriptor.bytes
                                             : descriptor.request_bytes[index];
}

template <bool EnableAsfe>
__device__ void direct_read_owner_loop(const PersistentKernelParams& params,
                                       u32 queue_count, u32 owner_block) {
#ifdef DVSTOR_HAVE_GPUNETIO
  constexpr u32 warp_width = 32;
  constexpr u32 max_warps_per_block = 8;
  // At most eight same-priority descriptors are collected per service
  // boundary. Split tails reuse the original descriptor plus compact
  // count/offset arrays; keeping a second full descriptor matrix here would
  // make the unified owner/query kernel exceed the 48-KiB static shared-memory
  // contract and would reduce query residency.
  constexpr u32 max_submit_batches = 8;
  const u32 lane = threadIdx.x % warp_width;
  const u32 warps_per_block = blockDim.x / warp_width;
  const u32 warp_in_block = threadIdx.x / warp_width;
  const u32 warp = owner_block * warps_per_block + warp_in_block;
  if (warps_per_block == 0 || warps_per_block > max_warps_per_block ||
      warp >= queue_count)
    return;
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

  __shared__ DirectBatchDescriptor
    shared_batches[max_warps_per_block][max_submit_batches];
  __shared__ u16 shared_wqe_offsets[max_warps_per_block][max_submit_batches];
  __shared__ u16
    shared_tail_matching_counts[max_warps_per_block][max_submit_batches];
  __shared__ u16
    shared_tail_wqe_offsets[max_warps_per_block][max_submit_batches];
  __shared__ u32 shared_batch_counts[max_warps_per_block];
  __shared__ u32 shared_total_wqes[max_warps_per_block];
  __shared__ u32 shared_total_tail_wqes[max_warps_per_block];

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
  const DeviceRingView<DirectBatchDescriptor> critical_queue =
    params.direct_batch_queues[warp];
  const DeviceRingView<DirectBatchDescriptor> speculative_queue =
    params.direct_speculative_batch_queues == nullptr
      ? DeviceRingView<DirectBatchDescriptor>{}
      : params.direct_speculative_batch_queues[warp];
  DirectBatchDescriptor deferred{};
  bool have_deferred = false;
  bool trace_first_batch = true;
  DirectOwnerProgress* owner_progress = params.direct_owner_progress == nullptr
                                          ? nullptr
                                          : params.direct_owner_progress + warp;
  u64 last_heartbeat_ns = 0;

  if (lane == 0 && params.direct_owner_phases != nullptr) {
    params.direct_owner_phases[warp] = 1;
    __threadfence_system();
  }

  const u32 initial_idle_cycles = 256u + ((warp * 97u) & 2047u);
  u32 idle_cycles = initial_idle_cycles;
  for (;;) {
    const u32 stop_requested =
      lane == 0 ? *reinterpret_cast<const volatile u32*>(params.stop) : 0u;
    if (__shfl_sync(0xffffffffu, stop_requested, 0) != 0) break;

    if (lane == 0) {
      if (owner_progress != nullptr) {
        const u64 announced =
          *reinterpret_cast<const volatile u64*>(&owner_progress->announced);
        const u64 completed =
          *reinterpret_cast<const volatile u64*>(&owner_progress->completed);
        const u64 now_ns = global_time_ns();
        const u64 half_timeout_ns = params.direct_timeout_ns / 2;
        const u64 heartbeat_period_ns =
          half_timeout_ns < u64{1'000'000} ? u64{1'000'000} : half_timeout_ns;
        if (announced != completed &&
            now_ns - last_heartbeat_ns >= heartbeat_period_ns) {
          record_owner_watchdog_counter(&owner_progress->heartbeat);
          last_heartbeat_ns = now_ns;
        }
      }
      u32 batch_count = 0;
      u32 total_wqes = 0;
      DirectBatchPriority submission_priority = DirectBatchPriority::critical;
      while (batch_count < max_submit_batches) {
        // One standalone speculative descriptor is the complete idle-service
        // quantum.  Bounding it here, rather than by a width magic number in
        // the query CTA, guarantees that a later critical arrival can wait
        // behind at most one already-posted speculative CQ boundary.
        if (batch_count != 0 &&
            submission_priority == DirectBatchPriority::speculative) {
          break;
        }
        DirectBatchDescriptor descriptor{};
        u32 matching = 0;
        u32 tail_matching = 0;
        if (have_deferred) {
          descriptor = deferred;
          have_deferred = false;
          submission_priority =
            static_cast<DirectBatchPriority>(descriptor.priority);
        } else {
          // Strict critical-first service: an idle owner checks the critical
          // ring before considering the disjoint speculative ring.  Once a
          // critical train is being collected this boundary never switches
          // priority; standalone speculation is admitted only when no
          // critical descriptor was visible.
          bool popped = device_ring_try_pop(critical_queue, descriptor);
          if (!popped && batch_count == 0) {
            if constexpr (EnableAsfe) {
              if (params.direct_speculative_batch_queues != nullptr) {
                popped = device_ring_try_pop(speculative_queue, descriptor);
                if (popped) {
                  submission_priority = DirectBatchPriority::speculative;
                }
              }
            }
          }
          if (!popped) {
            break;
          }
          if (owner_progress != nullptr) {
            record_owner_watchdog_counter(&owner_progress->dequeued);
          }
          if (trace_first_batch && params.direct_owner_phases != nullptr) {
            params.direct_owner_phases[warp] = 2;
            __threadfence_system();
          }
        }

        const u32 split = descriptor.critical_request_count;
        const bool mandatory_fenced_tail = is_mandatory_fenced_tail(descriptor);
        const bool mixed_mandatory_tail =
          is_mixed_mandatory_fenced_tail(descriptor);
        const bool exact_snapshot_tail =
          (descriptor.flags & kDirectBatchFlagMandatoryFencedTail) != 0;
        // A one-CQE exact snapshot train is an indivisible owner submission:
        // if ordinary critical work was already collected, defer the train;
        // if the train is first, stop collecting after it. This prevents an
        // unrelated descriptor from requiring a prefix CQE inside
        // [full READs][fenced trailers].
        if (mandatory_fenced_tail && batch_count != 0) {
          deferred = descriptor;
          have_deferred = true;
          break;
        }
        const bool split_valid =
          (split == 0 && descriptor.flags == 0 &&
           descriptor.completion_status != nullptr &&
           descriptor.speculative_completion_status == nullptr &&
           descriptor.speculative_completion_timestamp_ns == nullptr) ||
          (EnableAsfe && !mandatory_fenced_tail && split != 0 &&
           descriptor.completion_status != nullptr &&
           descriptor.priority ==
             static_cast<u8>(DirectBatchPriority::critical) &&
           split < descriptor.request_count &&
           descriptor.speculative_completion_status != nullptr) ||
          (mandatory_fenced_tail && split != 0 &&
           exact_snapshot_tail != mixed_mandatory_tail &&
           descriptor.completion_status != nullptr &&
           descriptor.priority ==
             static_cast<u8>(DirectBatchPriority::critical) &&
           ((!mixed_mandatory_tail && descriptor.request_count == 2u * split) ||
            (mixed_mandatory_tail && descriptor.request_count >= 2u * split)) &&
           descriptor.request_bytes == nullptr &&
           descriptor.speculative_completion_status == nullptr &&
           descriptor.speculative_completion_timestamp_ns == nullptr &&
           (descriptor.flags & ~kDirectBatchKnownFlags) == 0);
        if (!split_valid) {
          complete_direct_batch_with_unadmitted_tail(descriptor, -EINVAL,
                                                     owner_progress);
          continue;
        }
        if (descriptor.memory_node == memory_node &&
            descriptor.request_shards != nullptr &&
            descriptor.remote_offsets != nullptr &&
            descriptor.local_iova_offsets != nullptr && descriptor.bytes != 0 &&
            descriptor.bytes <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
          bool request_lengths_valid = true;
          for (u32 index = 0; index < descriptor.request_count; ++index) {
            if (descriptor.request_shards[index] != memory_node) continue;
            if (mandatory_fenced_tail || (EnableAsfe && split != 0)) {
              if (index >= split)
                ++tail_matching;
              else
                ++matching;
            } else {
              ++matching;
            }
            const u32 request_length =
              direct_batch_request_length(descriptor, index);
            request_lengths_valid &=
              request_length != 0 && request_length <= descriptor.bytes &&
              request_length <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE;
          }
          if (mandatory_fenced_tail) {
            for (u32 index = 0; index < split; ++index) {
              request_lengths_valid &=
                descriptor.request_shards[index] ==
                  descriptor.request_shards[split + index] &&
                descriptor.remote_offsets[index] ==
                  descriptor.remote_offsets[split + index] &&
                exact_snapshot_local_layout_matches(
                  descriptor.local_iova_offsets[index],
                  descriptor.local_iova_offsets[split + index],
                  descriptor.bytes);
            }
          }
          if (!request_lengths_valid) {
            matching = 0;
            tail_matching = 0;
          }
        }
        const bool split_groups_valid =
          split == 0 || (matching != 0 && tail_matching != 0 &&
                         (!mandatory_fenced_tail ||
                          (mixed_mandatory_tail ? tail_matching >= matching
                                                : tail_matching == matching)));
        if (descriptor.priority != static_cast<u8>(submission_priority)) {
          complete_direct_batch_with_unadmitted_tail(descriptor, -EINVAL,
                                                     owner_progress);
          continue;
        }
        if (!split_groups_valid || descriptor.memory_node != memory_node ||
            matching == 0 || descriptor.bytes == 0 ||
            descriptor.bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
          complete_direct_batch_with_unadmitted_tail(descriptor, -EINVAL,
                                                     owner_progress);
          continue;
        }
        if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) !=
            0) {
          complete_direct_batch_with_unadmitted_tail(descriptor, -EHOSTDOWN,
                                                     owner_progress);
          continue;
        }
        // All descriptors collected below are published by one SQ doorbell.
        // Reserve at most one dump WQE for that submission, rather than one
        // per logical descriptor. RC execution is ordered and the single
        // final signaled WQE covers every preceding read.
        const u32 needed =
          matching + (mandatory_fenced_tail ? tail_matching : 0u);
        // A mandatory train has only the final CQ boundary. Older hardware
        // that requires dump WQEs therefore needs one final dump, not the two
        // dumps used by the independently completed ASFE prefix/tail pair.
        const u32 completion_wqes = need_dump ? 1u : 0u;
        if (needed + completion_wqes > qp->sq_wqe_num) {
          complete_direct_batch_with_unadmitted_tail(descriptor, -E2BIG,
                                                     owner_progress);
          continue;
        }
        if (batch_count != 0 &&
            total_wqes + needed + completion_wqes > qp->sq_wqe_num) {
          deferred = descriptor;
          have_deferred = true;
          break;
        }
        shared_batches[warp_in_block][batch_count] = descriptor;
        shared_wqe_offsets[warp_in_block][batch_count] =
          static_cast<u16>(total_wqes);
        shared_tail_matching_counts[warp_in_block][batch_count] =
          static_cast<u16>(tail_matching);
        shared_tail_wqe_offsets[warp_in_block][batch_count] = 0;
        ++batch_count;
        total_wqes += matching;
        if (mandatory_fenced_tail) {
          // Exact trailers are admitted with their prefix above, never from
          // speculative slack. Keep the full-record WQE count separate
          // because the common layout emits all suffixes after it.
          break;
        }
      }

      // A split descriptor may append its shadow suffix to the same SQ train
      // only after every critical descriptor visible at this service
      // boundary has been collected. The first fence publishes all critical
      // completions; the second publishes only the admitted suffix. If a
      // critical descriptor was deferred, the collection saturated, or the
      // suffix does not fit, that suffix is rejected fail-soft. It is never
      // forwarded to another QP because doing so would create a marginal
      // speculative doorbell and could delay subsequently arriving critical
      // work.
      const bool can_steal_sq_slack =
        submission_priority == DirectBatchPriority::critical &&
        !have_deferred && batch_count < max_submit_batches;
      u32 total_tail_wqes = 0;
      for (u32 batch = 0; batch < batch_count; ++batch) {
        const u32 candidate_tail =
          shared_tail_matching_counts[warp_in_block][batch];
        if (candidate_tail == 0) continue;
        const bool mandatory_fenced_tail =
          is_mandatory_fenced_tail(shared_batches[warp_in_block][batch]);
        if (mandatory_fenced_tail) {
          shared_tail_wqe_offsets[warp_in_block][batch] =
            static_cast<u16>(total_tail_wqes);
          total_tail_wqes += candidate_tail;
          continue;
        }
        if constexpr (EnableAsfe) {
          const u32 critical_fence_wqes = need_dump ? 1u : 0u;
          const u32 tail_fence_wqes = need_dump ? 1u : 0u;
          const bool fits = can_steal_sq_slack &&
                            total_wqes + critical_fence_wqes + total_tail_wqes +
                                candidate_tail + tail_fence_wqes <=
                              qp->sq_wqe_num;
          if (fits) {
            shared_tail_wqe_offsets[warp_in_block][batch] =
              static_cast<u16>(total_tail_wqes);
            total_tail_wqes += candidate_tail;
            if (owner_progress != nullptr) {
              // direct_fetch_split_batch announced the critical descriptor.
              // Track the independently published second completion as one
              // additional logical owner operation.
              record_owner_watchdog_counter(&owner_progress->announced);
            }
          } else {
            shared_tail_matching_counts[warp_in_block][batch] = 0;
            shared_tail_wqe_offsets[warp_in_block][batch] = 0;
            complete_unadmitted_speculative_tail(
              shared_batches[warp_in_block][batch], -EAGAIN);
          }
        }
      }
      shared_batch_counts[warp_in_block] = batch_count;
      shared_total_wqes[warp_in_block] = total_wqes;
      shared_total_tail_wqes[warp_in_block] = total_tail_wqes;
    }
    __syncwarp();

    const u32 batch_count = shared_batch_counts[warp_in_block];
    if (batch_count == 0) {
      if (lane == 0) {
        device_ring_relax(idle_cycles);
      }
      __syncwarp();
      idle_cycles = min(idle_cycles * 2, 16384u);
      continue;
    }
    idle_cycles = initial_idle_cycles;
    const bool mandatory_snapshot_train =
      batch_count == 1 &&
      is_mandatory_fenced_tail(shared_batches[warp_in_block][0]);

    const doca_gpu_dev_verbs_ticket_t first_wqe = qp->sq_wqe_pi;
    const doca_gpu_dev_verbs_ticket_t first_completion =
      doca_gpu_dev_verbs_load_relaxed<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
        &completion_queue->cqe_ci);
    for (u32 batch = 0; batch < batch_count; ++batch) {
      const DirectBatchDescriptor descriptor =
        shared_batches[warp_in_block][batch];
      const u32 batch_offset = shared_wqe_offsets[warp_in_block][batch];
      const u32 critical_count =
        (is_mandatory_fenced_tail(descriptor) ||
         (EnableAsfe && descriptor.critical_request_count != 0))
          ? static_cast<u32>(descriptor.critical_request_count)
          : static_cast<u32>(descriptor.request_count);
      u32 matched_before = 0;
      for (u32 base = 0; base < critical_count; base += warp_width) {
        const u32 index = base + lane;
        const bool matching_request =
          index < critical_count &&
          descriptor.request_shards[index] == memory_node;
        const u32 matching_mask = __ballot_sync(0xffffffffu, matching_request);
        const u32 request_length = matching_request
          ? direct_batch_request_length(descriptor, index) : 0u;
        const bool remote_range_valid = !matching_request ||
          direct_remote_range_valid(
            region.bytes, descriptor.remote_offsets[index], request_length);
        const u32 invalid_mask = __ballot_sync(
          0xffffffffu, matching_request && !remote_range_valid);
        if (lane == 0 && invalid_mask != 0) {
          // batch_count was copied into a register above and is never loaded
          // from shared memory again. Reuse its otherwise-dead high bit for
          // the pre-doorbell range error instead of extending a predicate's
          // live range across the complete WQE construction loop.
          shared_batch_counts[warp_in_block] |= 0x80000000u;
        }
        if (matching_request && remote_range_valid) {
          const u32 lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
          const u32 rank = __popc(matching_mask & lower_lanes);
          const u32 matched = matched_before + rank;
          const doca_gpu_dev_verbs_ticket_t ticket =
            first_wqe + batch_offset + matched;
          auto* wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, ticket);
          const bool final_critical_read =
            batch_offset + matched + 1u == shared_total_wqes[warp_in_block];
          const auto flags =
            !mandatory_snapshot_train && !need_dump && final_critical_read
              ? DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE
              : DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;
          doca_gpu_dev_verbs_wqe_prepare_read(
            qp, wqe, ticket, flags,
            region.address + descriptor.remote_offsets[index], region.rkey,
            descriptor.local_iova_offsets[index], params.direct_local_mkey,
            request_length);
        }
        matched_before += __popc(matching_mask);
      }
    }
    __syncwarp();

    const u32 critical_read_wqes = shared_total_wqes[warp_in_block];
    const u32 tail_read_wqes = shared_total_tail_wqes[warp_in_block];
    const u32 tail_wqe_base =
      critical_read_wqes + (need_dump && !mandatory_snapshot_train ? 1u : 0u);
    if (tail_read_wqes != 0) {
      for (u32 batch = 0; batch < batch_count; ++batch) {
        const DirectBatchDescriptor descriptor =
          shared_batches[warp_in_block][batch];
        const u32 matching = shared_tail_matching_counts[warp_in_block][batch];
        if (matching == 0) continue;
        const u32 batch_offset = shared_tail_wqe_offsets[warp_in_block][batch];
        const u32 split = descriptor.critical_request_count;
        u32 matched_before = 0;
        for (u32 base = split; base < descriptor.request_count;
             base += warp_width) {
          const u32 index = base + lane;
          const bool matching_request =
            index < descriptor.request_count &&
            descriptor.request_shards[index] == memory_node;
          const u32 matching_mask =
            __ballot_sync(0xffffffffu, matching_request);
          const u32 request_length = matching_request
            ? direct_batch_request_length(descriptor, index) : 0u;
          const bool remote_range_valid = !matching_request ||
            direct_remote_range_valid(
              region.bytes, descriptor.remote_offsets[index], request_length);
          const u32 invalid_mask = __ballot_sync(
            0xffffffffu, matching_request && !remote_range_valid);
          if (lane == 0 && invalid_mask != 0) {
            shared_batch_counts[warp_in_block] |= 0x80000000u;
          }
          if (matching_request && remote_range_valid) {
            const u32 lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
            const u32 rank = __popc(matching_mask & lower_lanes);
            const u32 matched = matched_before + rank;
            const doca_gpu_dev_verbs_ticket_t ticket =
              first_wqe + tail_wqe_base + batch_offset + matched;
            auto* wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, ticket);
            const bool final_tail_read =
              batch_offset + matched + 1u == tail_read_wqes;
            const bool first_tail_read = batch_offset + matched == 0u;
            const u32 completion_flags =
              !need_dump && final_tail_read
                ? DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE
                : DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;
            // mlx5 FENCE_AND_INITIATOR_SMALL_FENCE on the first trailer
            // prevents any trailer READ from passing an outstanding full
            // record READ. The final CQE therefore proves completion of both
            // snapshots without a query-side round trip.
            const auto flags = static_cast<doca_gpu_dev_verbs_wqe_ctrl_flags>(
              completion_flags | (mandatory_snapshot_train && first_tail_read
                                    ? DOCA_GPUNETIO_MLX5_WQE_CTRL_FENCE
                                    : 0u));
            doca_gpu_dev_verbs_wqe_prepare_read(
              qp, wqe, ticket, flags,
              region.address + descriptor.remote_offsets[index], region.rkey,
              descriptor.local_iova_offsets[index], params.direct_local_mkey,
              request_length);
          }
          matched_before += __popc(matching_mask);
        }
      }
    }
    __syncwarp();

    // Range validation is deliberately fused into the warp-parallel WQE
    // preparation pass above.  The owner used to repeat every 64-bit bounds
    // check serially in lane 0 while counting the descriptor, extending the
    // critical owner/CQ path for all valid queries.  A bad address leaves its
    // WQE slot unprepared and the whole SQ train is rejected before the
    // doorbell, so no invalid READ can reach the NIC.  Complete independently
    // announced ASFE tails as well or the owner watchdog would retain debt.
    if ((shared_batch_counts[warp_in_block] & 0x80000000u) != 0) {
      if (lane == 0) {
        for (u32 batch = 0; batch < batch_count; ++batch) {
          const DirectBatchDescriptor descriptor =
            shared_batches[warp_in_block][batch];
          complete_direct_batch(descriptor, -ERANGE, owner_progress);
          if constexpr (EnableAsfe) {
            if (!is_mandatory_fenced_tail(descriptor) &&
                shared_tail_matching_counts[warp_in_block][batch] != 0) {
              complete_direct_batch(
                make_speculative_tail_descriptor(descriptor), -ERANGE,
                owner_progress);
            }
          }
        }
      }
      __syncwarp();
      continue;
    }

    if (lane == 0) {
      const u32 critical_fence_wqes =
        need_dump && !mandatory_snapshot_train ? 1u : 0u;
      const u32 tail_fence_wqes = need_dump && tail_read_wqes != 0 ? 1u : 0u;
      const u32 submission_wqes = critical_read_wqes + critical_fence_wqes +
                                  tail_read_wqes + tail_fence_wqes;
      if (owner_progress != nullptr) {
        u32 critical_batches = 0;
        u32 speculative_batches = 0;
        for (u32 batch = 0; batch < batch_count; ++batch) {
          if (shared_batches[warp_in_block][batch].priority ==
              static_cast<u8>(DirectBatchPriority::speculative)) {
            ++speculative_batches;
          } else {
            ++critical_batches;
          }
          if constexpr (EnableAsfe) {
            if (shared_tail_matching_counts[warp_in_block][batch] != 0) {
              if (is_mandatory_fenced_tail(
                    shared_batches[warp_in_block][batch])) {
                // One mandatory train has one completion and is wholly
                // correctness-critical.
              } else {
                ++speculative_batches;
              }
            }
          }
        }
        atomicAdd(&owner_progress->submitted_wqes,
                  static_cast<unsigned long long>(submission_wqes));
        atomicAdd(&owner_progress->submission_wqe_capacity,
                  static_cast<unsigned long long>(qp->sq_wqe_num));
        atomicAdd(&owner_progress->critical_batches,
                  static_cast<unsigned long long>(critical_batches));
        atomicAdd(&owner_progress->speculative_batches,
                  static_cast<unsigned long long>(speculative_batches));
      }
      if (need_dump && !mandatory_snapshot_train) {
        const doca_gpu_dev_verbs_ticket_t dump_ticket =
          first_wqe + critical_read_wqes;
        auto* dump_wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, dump_ticket);
        doca_gpu_dev_verbs_wqe_prepare_dump(
          qp, dump_wqe, dump_ticket, DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
          reinterpret_cast<u64>(params.direct_dump) -
            params.direct_local_iova_base,
          params.direct_local_mkey, 1);
        if constexpr (EnableAsfe) {
          if (tail_read_wqes != 0) {
            const doca_gpu_dev_verbs_ticket_t tail_dump_ticket =
              first_wqe + tail_wqe_base + tail_read_wqes;
            auto* tail_dump_wqe =
              doca_gpu_dev_verbs_get_wqe_ptr(qp, tail_dump_ticket);
            doca_gpu_dev_verbs_wqe_prepare_dump(
              qp, tail_dump_wqe, tail_dump_ticket,
              DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
              reinterpret_cast<u64>(params.direct_dump) -
                params.direct_local_iova_base,
              params.direct_local_mkey, 1);
          }
        }
      }
      if (need_dump && mandatory_snapshot_train) {
        const doca_gpu_dev_verbs_ticket_t final_dump_ticket =
          first_wqe + tail_wqe_base + tail_read_wqes;
        auto* final_dump_wqe =
          doca_gpu_dev_verbs_get_wqe_ptr(qp, final_dump_ticket);
        doca_gpu_dev_verbs_wqe_prepare_dump(
          qp, final_dump_wqe, final_dump_ticket,
          DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
          reinterpret_cast<u64>(params.direct_dump) -
            params.direct_local_iova_base,
          params.direct_local_mkey, 1);
      }
      if (trace_first_batch && params.direct_owner_phases != nullptr) {
        params.direct_owner_phases[warp] = 3;
        __threadfence_system();
      }
      doca_gpu_dev_verbs_submit<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
        qp, first_wqe + submission_wqes);
      if (trace_first_batch && params.direct_owner_phases != nullptr) {
        params.direct_owner_phases[warp] = 4;
        __threadfence_system();
      }

      // Ordinary split batches expose a critical-prefix CQE followed by an
      // optional-tail CQE. A mandatory exact snapshot train is deliberately
      // isolated and has only this one, final CQE; the hardware fence on its
      // first trailer orders the two snapshots without a query-side wait.
      const i32 critical_status = poll_direct_cq(
        completion_queue, first_completion, params.direct_timeout_ns,
        params.stop, params.direct_disabled);
      if (critical_status == -ETIMEDOUT) {
        auto* completion_base = reinterpret_cast<mlx5_cqe64*>(
          __ldg(reinterpret_cast<uintptr_t*>(&completion_queue->cqe_daddr)));
        const u32 completion_count = __ldg(&completion_queue->cqe_num);
        const u32 completion_index =
          static_cast<u32>(first_completion) & (completion_count - 1u);
        const u64 observed_consumer = doca_gpu_dev_verbs_load_relaxed<
          DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
          &completion_queue->cqe_ci);
        const u8 observed_owner = doca_gpu_dev_verbs_load_relaxed_sys_global(
          reinterpret_cast<u8*>(&completion_base[completion_index].op_own));
        const u32 observed_dbrec = doca_gpu_dev_verbs_bswap32(
          *reinterpret_cast<const volatile u32*>(completion_queue->dbrec));
        const DirectBatchDescriptor first_descriptor =
          shared_batches[warp_in_block][0];
        u64 first_remote_offset = 0;
        u64 first_local_iova = 0;
        u32 first_request_bytes = first_descriptor.bytes;
        for (u32 request = 0; request < first_descriptor.request_count;
             ++request) {
          if (first_descriptor.request_shards[request] != memory_node) {
            continue;
          }
          first_remote_offset = first_descriptor.remote_offsets[request];
          if (first_descriptor.local_iova_offsets != nullptr) {
            first_local_iova = first_descriptor.local_iova_offsets[request];
          }
          first_request_bytes =
            direct_batch_request_length(first_descriptor, request);
          break;
        }
        printf(
          "[gpu-search] direct CQ timeout owner=%u node=%u batches=%u "
          "critical_reads=%u tail_reads=%u dump=%u "
          "first_wqe=%llu sq_pi=%llu cq_ticket=%llu "
          "cq_ci=%llu cq_dbrec=%u cq_index=%u cq_count=%u "
          "op_own=0x%x bytes=%u "
          "remote_offset=%llu local_iova=%llu\n",
          warp, memory_node, batch_count, critical_read_wqes, tail_read_wqes,
          need_dump ? 1u : 0u, static_cast<unsigned long long>(first_wqe),
          static_cast<unsigned long long>(qp->sq_wqe_pi),
          static_cast<unsigned long long>(first_completion),
          static_cast<unsigned long long>(observed_consumer), observed_dbrec,
          completion_index, completion_count,
          static_cast<unsigned>(observed_owner), first_request_bytes,
          static_cast<unsigned long long>(first_remote_offset),
          static_cast<unsigned long long>(first_local_iova));
      }
      // Capture the physical CQ boundary before owner-side validation so RDMA
      // completion telemetry does not charge checksum work as network delay.
      u64 critical_completion_ns = global_time_ns();
      if (mandatory_snapshot_train) {
        complete_direct_batch(shared_batches[warp_in_block][0], critical_status,
                              owner_progress, &critical_completion_ns);
      } else {
        for (u32 batch = 0; batch < batch_count; ++batch) {
          const DirectBatchDescriptor descriptor =
            shared_batches[warp_in_block][batch];
          complete_direct_batch(descriptor, critical_status, owner_progress,
                                &critical_completion_ns);
        }
      }

      i32 tail_status = critical_status;
      u64 tail_completion_ns = critical_completion_ns;
      if constexpr (EnableAsfe) {
        if (!mandatory_snapshot_train && critical_status == 0 &&
            tail_read_wqes != 0) {
          tail_status = poll_direct_cq(completion_queue, first_completion + 1u,
                                       params.direct_timeout_ns, params.stop,
                                       params.direct_disabled);
          tail_completion_ns = global_time_ns();
          if (tail_status == -ETIMEDOUT) {
            printf(
              "[gpu-search] speculative CQ timeout owner=%u node=%u "
              "tail_reads=%u cq_ticket=%llu\n",
              warp, memory_node, tail_read_wqes,
              static_cast<unsigned long long>(first_completion + 1u));
          }
        }
        if (!mandatory_snapshot_train && tail_read_wqes != 0) {
          for (u32 batch = 0; batch < batch_count; ++batch) {
            if (shared_tail_matching_counts[warp_in_block][batch] == 0) {
              continue;
            }
            complete_direct_batch(make_speculative_tail_descriptor(
                                    shared_batches[warp_in_block][batch]),
                                  tail_status, owner_progress,
                                  &tail_completion_ns);
          }
        }
      }

      const i32 submission_status =
        critical_status != 0 ? critical_status : tail_status;
      if (trace_first_batch && params.direct_owner_phases != nullptr) {
        params.direct_owner_phases[warp] = submission_status == 0 ? 6u : 5u;
        __threadfence_system();
        trace_first_batch = false;
      }
      if (submission_status != 0 && submission_status != -ECANCELED) {
        // An error CQE may precede the two explicit fences and leave later
        // flush CQEs unconsumed. Enter transport-wide fail-stop immediately:
        // lifecycle recovery must destroy/recreate the QP and CQ, never
        // clear direct_disabled and reuse their existing producer/consumer
        // tickets in place.
        if (params.direct_error != nullptr) {
          atomicCAS(params.direct_error, 0, submission_status);
        }
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
  direct_read_owner_loop<true>(params, queue_count, blockIdx.x);
}

__global__ void gpunetio_locked_read_probe_kernel(PersistentKernelParams params,
                                                  u8* destinations,
                                                  u32 destination_stride,
                                                  i32* statuses, u32* completed,
                                                  u32 iterations) {
  constexpr u32 warp_width = 32;
  if (threadIdx.x % warp_width != 0) return;
  const u32 worker = threadIdx.x / warp_width;
  const u32 worker_count =
    min(params.direct_qps_per_node, blockDim.x / warp_width);
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

__global__ void gpunetio_batched_read_probe_kernel(
  PersistentKernelParams params, u8* destinations, u32 destination_stride,
  i32* statuses, u32* completed, u32 batch_size) {
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
      destinations +
        static_cast<size_t>(blockIdx.x) * batch_size * destination_stride,
      destination_stride, sizeof(u64), blockIdx.x % params.direct_qps_per_node);
    if (status == 0) {
      status =
        direct_fetch(params, memory_node, 0,
                     destinations + static_cast<size_t>(blockIdx.x) *
                                      batch_size * destination_stride,
                     sizeof(u64), blockIdx.x % params.direct_qps_per_node);
    }
    statuses[blockIdx.x] = status;
    if (status == 0) atomicAdd(completed, batch_size + 1);
  }
}

__global__ void gpunetio_owner_read_probe_kernel(
  PersistentKernelParams params, u32* request_shards, u64* remote_offsets,
  u64* local_iova_offsets, u8* destinations, u32 destination_stride,
  i32* statuses, u32* completed, u32* phases, u32 queue_count) {
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
    params, memory_node, request_shards + qp_index, remote_offsets + qp_index,
    1, destinations + static_cast<size_t>(qp_index) * destination_stride,
    destination_stride, sizeof(u64), lane, local_iova_offsets + qp_index,
    completion_status, false, phases + qp_index);
  statuses[qp_index] = status;
  if (status == 0) atomicAdd(completed, 1u);
  __threadfence_system();
  phases[qp_index] = 4;
  __threadfence_system();
}

}  // namespace gpu_search::persistent_kernel_detail
