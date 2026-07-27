#pragma once

#include "gpu_search/persistent_kernel/query_traversal.cuh"

namespace gpu_search::persistent_kernel_detail {

__device__ void direct_read_owner_loop(PersistentKernelParams params,
                                       u32 queue_count,
                                       u32 owner_block);

template <bool EnableAdjacencyOracle>
__global__ void persistent_search_kernel(PersistentKernelParams params) {
  const bool unified_dispatch = params.direct_owner_block_count != 0;
  if (unified_dispatch && blockIdx.x < params.direct_owner_block_count) {
    direct_read_owner_loop(params, params.direct_batch_queue_count, blockIdx.x);
    return;
  }

  bool enable_queries = true;
  bool enable_dispatcher = false;
  bool enable_route_control = true;
  if (unified_dispatch) {
    const u32 role_block = blockIdx.x - params.direct_owner_block_count;
    enable_queries = role_block < params.query_block_count;
    enable_dispatcher = role_block == params.query_block_count;
    enable_route_control = role_block == params.query_block_count + 1;
    if (!enable_queries && !enable_dispatcher && !enable_route_control) return;
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
  __shared__ CentroidRoutePublishDescriptor route_descriptor;
  __shared__ u32 have_submission;
  __shared__ u32 dispatch_pending;
  __shared__ u32 have_route_submission;
  __shared__ u32 stop_requested;
  __shared__ u32 idle_cycles;
  __shared__ i32 route_status;
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
      have_route_submission = enable_route_control &&
        params.route_submissions.entries != nullptr &&
        device_ring_try_pop(params.route_submissions, route_descriptor)
          ? 1u : 0u;
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
        for (u32 index = threadIdx.x;
             index < route_descriptor.update_count;
             index += blockDim.x) {
          const CentroidRouteUpdate update =
            params.centroid_route_updates[index];
          bool duplicate_shard = false;
          for (u32 prior = 0; prior < index; ++prior) {
            duplicate_shard = duplicate_shard ||
              params.centroid_route_updates[prior].shard == update.shard;
          }
          bool invalid_entry = false;
          for (u32 entry = 0; entry < update.live_entry_count; ++entry) {
            const DeviceCentroidRouteEntry& candidate = update.entries[entry];
            invalid_entry = invalid_entry || candidate.remote_node == 0 ||
              candidate.flags != kCentroidRouteLive ||
              remote_shard(candidate.remote_node) != update.shard;
            for (u32 prior = 0; prior < entry; ++prior) {
              invalid_entry = invalid_entry ||
                update.entries[prior].remote_node == candidate.remote_node;
            }
          }
          if (update.shard >= params.num_shards ||
              update.shard >= params.centroid_route_shard_capacity ||
              update.version == 0 ||
              update.live_entry_count > params.centroid_route_entry_capacity ||
              ((update.vector_count == 0) !=
               (update.live_entry_count == 0)) ||
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
          const u64 previous = route_epoch.fetch_add(
            1, cuda::memory_order_acq_rel);
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
        for (u32 index = threadIdx.x;
             index < route_descriptor.update_count;
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
             item < static_cast<u64>(route_descriptor.update_count) *
                      params.dim;
             item += blockDim.x) {
          const u32 update_index = static_cast<u32>(item / params.dim);
          const u32 dimension = static_cast<u32>(item % params.dim);
          const CentroidRouteUpdate update =
            params.centroid_route_updates[update_index];
          params.shard_centroids[
            static_cast<size_t>(update.shard) * params.dim + dimension] =
              params.centroid_route_centroid_updates[item];
        }
        for (u64 item = threadIdx.x;
             item < static_cast<u64>(route_descriptor.update_count) *
                      params.centroid_route_entry_capacity;
             item += blockDim.x) {
          const u32 update_index = static_cast<u32>(
            item / params.centroid_route_entry_capacity);
          const u32 entry = static_cast<u32>(
            item % params.centroid_route_entry_capacity);
          const CentroidRouteUpdate update =
            params.centroid_route_updates[update_index];
          params.centroid_route_entries[
            static_cast<size_t>(update.shard) *
              params.centroid_route_entry_capacity + entry] =
            entry < update.live_entry_count
              ? update.entries[entry] : DeviceCentroidRouteEntry{};
        }
        __threadfence();
        __syncthreads();

        for (u32 index = threadIdx.x;
             index < route_descriptor.update_count;
             index += blockDim.x) {
          const CentroidRouteUpdate update =
            params.centroid_route_updates[index];
          DeviceCentroidRouteShard& destination =
            params.centroid_route_shards[update.shard];
          cuda::atomic_ref<u64, cuda::thread_scope_device> sequence(
            destination.sequence);
          centroid_route_atomic_store(
            destination.command_id, route_descriptor.command_id);
          centroid_route_atomic_store(destination.version, update.version);
          centroid_route_atomic_store(
            destination.vector_count, update.vector_count);
          centroid_route_atomic_store(
            destination.live_entry_count, update.live_entry_count);
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
    process_query<EnableAdjacencyOracle>(params, descriptor);
    __syncthreads();
  }
}

__device__ void complete_direct_batch(const DirectBatchDescriptor& descriptor,
                                      i32 status,
                                      DirectOwnerProgress* owner_progress,
                                      u64* submission_completion_ns = nullptr) {
  if (descriptor.completion_status == nullptr) return;
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
  if (owner_progress != nullptr) {
    record_owner_watchdog_counter(&owner_progress->completed);
  }
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
  bool idle_credit_announced = false;
  bool trace_first_batch = true;
  DirectOwnerProgress* owner_progress = params.direct_owner_progress == nullptr
    ? nullptr : params.direct_owner_progress + warp;
  QpExpansionLeaseState* expansion_lease =
    params.expansion_qp_leases == nullptr ||
        warp >= params.expansion_qp_lease_count
      ? nullptr : params.expansion_qp_leases + warp;
  u64 last_heartbeat_ns = 0;

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
      if (owner_progress != nullptr) {
        const u64 announced = *reinterpret_cast<const volatile u64*>(
          &owner_progress->announced);
        const u64 completed = *reinterpret_cast<const volatile u64*>(
          &owner_progress->completed);
        const u64 now_ns = global_time_ns();
        const u64 half_timeout_ns = params.direct_timeout_ns / 2;
        const u64 heartbeat_period_ns = half_timeout_ns < u64{1'000'000}
          ? u64{1'000'000} : half_timeout_ns;
        if (announced != completed &&
            now_ns - last_heartbeat_ns >= heartbeat_period_ns) {
          record_owner_watchdog_counter(&owner_progress->heartbeat);
          last_heartbeat_ns = now_ns;
        }
      }
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
          if (owner_progress != nullptr) {
            record_owner_watchdog_counter(&owner_progress->dequeued);
          }
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
            bool request_lengths_valid = true;
            for (u32 index = 0; index < descriptor.request_count; ++index) {
              if (descriptor.request_shards[index] != memory_node) continue;
              ++matching;
              const u32 request_length = descriptor.request_bytes == nullptr
                ? descriptor.bytes : descriptor.request_bytes[index];
              request_lengths_valid &=
                request_length != 0 &&
                request_length <= descriptor.bytes &&
                request_length <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE;
            }
            if (!request_lengths_valid) matching = 0;
          }
        }

        if (descriptor.memory_node != memory_node || matching == 0 ||
            descriptor.bytes == 0 ||
            descriptor.bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
          complete_direct_batch(descriptor, -EINVAL, owner_progress);
          continue;
        }
        if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) != 0) {
          complete_direct_batch(descriptor, -EHOSTDOWN, owner_progress);
          continue;
        }
        // All descriptors collected below are published by one SQ doorbell.
        // Reserve at most one dump WQE for that submission, rather than one
        // per logical descriptor.  RC execution is ordered and the single
        // final signaled WQE covers every preceding read.
        const u32 needed = matching;
        const u32 completion_wqes = need_dump ? 1u : 0u;
        if (needed + completion_wqes > qp->sq_wqe_num) {
          complete_direct_batch(descriptor, -E2BIG, owner_progress);
          continue;
        }
        if (batch_count != 0 &&
            total_wqes + needed + completion_wqes > qp->sq_wqe_num) {
          expansion_pressure_clear_credit(
            params.expansion_pressure, false, true);
          qp_expansion_lease_revoke(expansion_lease);
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
      if (lane == 0) {
        const unsigned long long pressure =
          expansion_pressure_load(params.expansion_pressure);
        const u32 active_queries = expansion_pressure_active(pressure);
        bool progress_balanced = false;
        if (!have_deferred && owner_progress != nullptr) {
          const u64 announced = *reinterpret_cast<const volatile u64*>(
            &owner_progress->announced);
          const u64 completed = *reinterpret_cast<const volatile u64*>(
            &owner_progress->completed);
          progress_balanced = announced == completed;
        }
        if (expansion_owner_idle_episode_transition(
              active_queries, true, progress_balanced, false,
              idle_credit_announced)) {
          if (params.expansion_pressure != nullptr) {
            atomicAdd(
              &params.expansion_pressure->idle_owner_episodes, 1ULL);
          }
          const u32 completion_wqes = need_dump ? 1u : 0u;
          const u32 free_wqes = qp->sq_wqe_num > completion_wqes
            ? qp->sq_wqe_num - completion_wqes : 0u;
          (void)qp_expansion_lease_publish(
            expansion_lease, min(free_wqes, params.efficient_batch_cap));
        }
        device_ring_relax(idle_cycles);
      }
      __syncwarp();
      idle_cycles = min(idle_cycles * 2, 16384u);
      continue;
    }
    if (lane == 0) {
      qp_expansion_lease_revoke(expansion_lease);
      (void)expansion_owner_idle_episode_transition(
        expansion_pressure_active(
          expansion_pressure_load(params.expansion_pressure)),
        false, false, true, idle_credit_announced);
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
          const bool final_submission_read =
            batch + 1 == batch_count && matched + 1 == matching;
          const auto flags = !need_dump && final_submission_read
            ? DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE
            : DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;
          doca_gpu_dev_verbs_wqe_prepare_read(
            qp, wqe, ticket, flags,
            region.address + descriptor.remote_offsets[index], region.rkey,
            descriptor.local_iova_offsets[index], params.direct_local_mkey,
            descriptor.request_bytes == nullptr
              ? descriptor.bytes : descriptor.request_bytes[index]);
        }
        matched_before += __popc(matching_mask);
      }
    }
    __syncwarp();
    if (lane == 0) {
      const u32 read_wqes = shared_total_wqes[warp_in_block];
      const u32 submission_wqes = read_wqes + (need_dump ? 1u : 0u);
      if (need_dump) {
        const doca_gpu_dev_verbs_ticket_t dump_ticket = first_wqe + read_wqes;
        auto* dump_wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, dump_ticket);
        doca_gpu_dev_verbs_wqe_prepare_dump(
          qp, dump_wqe, dump_ticket, DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
          reinterpret_cast<u64>(params.direct_dump) -
            params.direct_local_iova_base,
          params.direct_local_mkey, 1);
      }
      if (trace_first_batch && params.direct_owner_phases != nullptr) {
        params.direct_owner_phases[warp] = 3;
        __threadfence_system();
      }
      doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
        qp, first_wqe + submission_wqes);
      if (trace_first_batch && params.direct_owner_phases != nullptr) {
        params.direct_owner_phases[warp] = 4;
        __threadfence_system();
      }

      // One final CQE is sufficient for the whole RC submission. Every
      // earlier read is ordered before it, while CQ_ERROR_UPDATE still emits
      // an immediate CQE if any intermediate WQE fails.
      const i32 status = poll_direct_cq(
        completion_queue, first_completion, params.direct_timeout_ns,
        params.stop, params.direct_disabled);
      if (status == -ETIMEDOUT) {
        auto* completion_base = reinterpret_cast<mlx5_cqe64*>(
          __ldg(reinterpret_cast<uintptr_t*>(&completion_queue->cqe_daddr)));
        const u32 completion_count = __ldg(&completion_queue->cqe_num);
        const u32 completion_index =
          static_cast<u32>(first_completion) & (completion_count - 1u);
        const u64 observed_consumer =
          doca_gpu_dev_verbs_load_relaxed<
            DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
              &completion_queue->cqe_ci);
        const u8 observed_owner =
          doca_gpu_dev_verbs_load_relaxed_sys_global(
            reinterpret_cast<u8*>(
              &completion_base[completion_index].op_own));
        const u32 observed_dbrec = doca_gpu_dev_verbs_bswap32(
          *reinterpret_cast<const volatile u32*>(completion_queue->dbrec));
        const DirectBatchDescriptor first_descriptor =
          shared_batches[warp_in_block][0];
        u64 first_remote_offset = 0;
        u64 first_local_iova = 0;
        u32 first_request_bytes = first_descriptor.bytes;
        for (u32 request = 0; request < first_descriptor.request_count;
             ++request) {
          if (first_descriptor.request_shards[request] != memory_node) continue;
          first_remote_offset = first_descriptor.remote_offsets[request];
          if (first_descriptor.local_iova_offsets != nullptr) {
            first_local_iova = first_descriptor.local_iova_offsets[request];
          }
          if (first_descriptor.request_bytes != nullptr) {
            first_request_bytes = first_descriptor.request_bytes[request];
          }
          break;
        }
        printf("[gpu-search] direct CQ timeout owner=%u node=%u batches=%u "
               "reads=%u dump=%u first_wqe=%llu sq_pi=%llu cq_ticket=%llu "
               "cq_ci=%llu cq_dbrec=%u cq_index=%u cq_count=%u "
               "op_own=0x%x bytes=%u "
               "remote_offset=%llu local_iova=%llu\n",
               warp, memory_node, batch_count, read_wqes,
               need_dump ? 1u : 0u,
               static_cast<unsigned long long>(first_wqe),
               static_cast<unsigned long long>(qp->sq_wqe_pi),
               static_cast<unsigned long long>(first_completion),
               static_cast<unsigned long long>(observed_consumer),
               observed_dbrec, completion_index, completion_count,
               static_cast<unsigned>(observed_owner), first_request_bytes,
               static_cast<unsigned long long>(first_remote_offset),
               static_cast<unsigned long long>(first_local_iova));
      }
      // All descriptors in this owner submission become software-visible at
      // one final-CQE boundary.  Reuse one timestamp so the trace does not
      // manufacture a completion spread from this publication loop itself.
      u64 submission_completion_ns = 0;
      for (u32 batch = 0; batch < batch_count; ++batch) {
        complete_direct_batch(shared_batches[warp_in_block][batch], status,
                              owner_progress, &submission_completion_ns);
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

}  // namespace gpu_search::persistent_kernel_detail
