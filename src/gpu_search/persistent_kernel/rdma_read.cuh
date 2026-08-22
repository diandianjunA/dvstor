#pragma once

#include <cuda/atomic>

#include "gpu_search/graph_record_validation.hh"
#include "gpu_search/persistent_kernel/candidate_scoring.cuh"
#include "vamana/dynamic_navigation_code.hh"

namespace gpu_search::persistent_kernel_detail {

__device__ __forceinline__ u32 dynamic_arena_state_load(
    const u32* state) {
  cuda::atomic_ref<u32, cuda::thread_scope_device> reference(
    *const_cast<u32*>(state));
  return reference.load(cuda::memory_order_acquire);
}

// Match atomicCAS's return-value convention while making the publication
// order explicit.  Successful reservations/extent transitions are acq_rel;
// failure returns an acquire observation for the caller's retry loop.
__device__ __forceinline__ u32 dynamic_arena_state_compare_exchange(
    u32* state, u32 expected, u32 desired) {
  cuda::atomic_ref<u32, cuda::thread_scope_device> reference(*state);
  u32 observed = expected;
  (void)reference.compare_exchange_strong(
    observed, desired, cuda::memory_order_acq_rel,
    cuda::memory_order_acquire);
  return observed;
}

__device__ __forceinline__ void dynamic_arena_state_publish(
    u32* state, u32 desired) {
  cuda::atomic_ref<u32, cuda::thread_scope_device> reference(*state);
  reference.store(desired, cuda::memory_order_release);
}

struct GraphFetchCycleBreakdown {
  u64 issue{};
  u64 wait{};
  u64 validation{};
  // Final-CQE owner submission-group latency. Keeping this separate from the
  // clock64 phase counters makes the signal available even when tracing is
  // disabled, without pretending that one CQE exposes per-WQE latency.
  u64 completion_latency_ns{};
  u64 completion_groups{};
};

#ifdef DVSTOR_HAVE_GPUNETIO
__device__ __forceinline__ void record_owner_watchdog_counter(
    unsigned long long* counter) {
  if (counter == nullptr) return;
  atomicAdd(counter, 1ULL);
}
#endif

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
  if (finite_f32_bits(distance) && distance < FLT_MAX) return distance;
  double wide_distance = 0.0;
  for (u32 dimension = 0; dimension < params.dim; ++dimension) {
    const double component = static_cast<double>(
      storage_component(params, vector, dimension));
    const double difference = static_cast<double>(query[dimension]) - component;
    wide_distance += difference * difference;
  }
  return saturate_device_squared_l2(wide_distance);
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
                                  u32* owner_progress = nullptr,
                                  u64* owner_completion_timestamp_ns = nullptr,
                                  const u32* request_bytes = nullptr,
                                  DirectBatchPriority priority =
                                    DirectBatchPriority::critical,
                                  bool nonblocking_enqueue = false,
                                  u32 prepared_matching = UINT32_MAX) {
#ifdef DVSTOR_HAVE_GPUNETIO
  u32 matching = prepared_matching;
  if (matching == UINT32_MAX) {
    matching = 0;
    for (u32 index = 0; index < request_count; ++index) {
      if (request_shards[index] == memory_node) ++matching;
    }
  }
  if (matching == 0) return 0;
  if (memory_node >= params.direct_region_count || params.direct_qps == nullptr ||
      params.direct_qp_locks == nullptr || params.direct_qps_per_node == 0 ||
      params.direct_disabled == nullptr ||
      *reinterpret_cast<volatile u32*>(params.direct_disabled) != 0) return -EHOSTDOWN;
  // Narrow-frontier callers already pass a normalized QP lane. Avoid a second
  // runtime integer remainder in every populated shard publication.
  u32 qp_lane = prepared_matching == UINT32_MAX
    ? lane % params.direct_qps_per_node : lane;
  const u32 qp_index = qp_lane *
    params.direct_region_count + memory_node;
  if (params.direct_qps[qp_index] == nullptr) return -EINVAL;
  // Critical and standalone speculative descriptors use disjoint rings.
  // The exclusive owner always drains the critical ring first and admits at
  // most one bounded speculative descriptor only at an otherwise idle
  // service boundary. This preserves critical WQE/CQ headroom without
  // forcing an early shadow read to wait for an unrelated critical prefix
  // solely to share its doorbell.
  const DeviceRingView<DirectBatchDescriptor>* selected_queues =
    priority == DirectBatchPriority::speculative
      ? params.direct_speculative_batch_queues
      : params.direct_batch_queues;
  if (priority == DirectBatchPriority::speculative &&
      selected_queues == nullptr) {
    if (owner_completion_timestamp_ns != nullptr) {
      *owner_completion_timestamp_ns = 0;
    }
    if (owner_completion != nullptr) {
      atomicExch(owner_completion, -EAGAIN);
    }
    return -EAGAIN;
  }
  if (selected_queues != nullptr && owner_completion != nullptr) {
    if (qp_index >= params.direct_batch_queue_count) return -EINVAL;
    DirectOwnerProgress* watchdog_progress =
      params.direct_owner_progress == nullptr
        ? nullptr : params.direct_owner_progress + qp_index;
    if (owner_progress != nullptr) {
      *reinterpret_cast<volatile u32*>(owner_progress) = 2;
      __threadfence_system();
    }
    if (prepared_matching == UINT32_MAX) {
      atomicExch(owner_completion, -EINPROGRESS);
    } else {
      // The narrow frontier wave has exclusive ownership of this completion
      // word until its batch is drained. Its caller also warp-synchronizes the
      // request SoA producers before entering this helper. The ring's
      // system-scope release below therefore publishes both that transitive
      // metadata and this initialization before an owner can consume the
      // descriptor, so a contended RMW and a separate fence are unnecessary.
      *reinterpret_cast<volatile i32*>(owner_completion) = -EINPROGRESS;
    }
    if (owner_completion_timestamp_ns != nullptr) {
      *owner_completion_timestamp_ns = 0;
    }
    if (prepared_matching == UINT32_MAX) {
      __threadfence();
    }
    const DirectBatchDescriptor descriptor{
      .request_shards = request_shards,
      .remote_offsets = remote_offsets,
      .local_iova_offsets = local_iova_offsets,
      .request_bytes = request_bytes,
      .completion_status = owner_completion,
      .completion_timestamp_ns = owner_completion_timestamp_ns,
      .request_count = static_cast<u16>(request_count),
      .memory_node = static_cast<u16>(memory_node),
      .bytes = bytes,
      .priority = static_cast<u8>(priority),
    };
    // Announce before attempting the bounded enqueue. This also covers an
    // owner that stopped before dequeueing enough entries to free a ring slot.
    // Cancellation before publication balances the monotonic counters below.
    if (watchdog_progress != nullptr) {
      record_owner_watchdog_counter(&watchdog_progress->announced);
    }
    bool pushed = device_ring_try_push(
      selected_queues[qp_index], descriptor);
    if (!pushed && nonblocking_enqueue) {
      atomicExch(owner_completion, -EAGAIN);
      if (watchdog_progress != nullptr) {
        record_owner_watchdog_counter(&watchdog_progress->completed);
      }
      return -EAGAIN;
    }
    while (!pushed) {
      if (*reinterpret_cast<const volatile u32*>(params.stop) != 0) {
        atomicExch(owner_completion, -ECANCELED);
        if (watchdog_progress != nullptr) {
          record_owner_watchdog_counter(&watchdog_progress->completed);
        }
        return -ECANCELED;
      }
      if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) != 0) {
        atomicExch(owner_completion, -EHOSTDOWN);
        if (watchdog_progress != nullptr) {
          record_owner_watchdog_counter(&watchdog_progress->completed);
        }
        return -EHOSTDOWN;
      }
      // A full descriptor ring is bounded backpressure, not evidence that the
      // QP has failed.  The owner warp is the sole WQE/CQ authority and its CQ
      // watchdog below is responsible for declaring a transport failure.
      device_ring_relax(128);
      pushed = device_ring_try_push(
        selected_queues[qp_index], descriptor);
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
      // Waiting behind already admitted bounded work is normal under load.
      // Only the owner that posts the WQE may turn a missing/error CQE into a
      // global fail-stop transition.
      device_ring_relax(128);
    }
  }
  auto* qp = reinterpret_cast<doca_gpu_dev_verbs_qp*>(params.direct_qps[qp_index]);
  if (bytes == 0) return -EINVAL;
  if (bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE ||
      matching + (qp->need_dump ? 1u : 0u) > qp->sq_wqe_num) return -E2BIG;
  for (u32 index = 0; index < request_count; ++index) {
    if (request_shards[index] != memory_node) continue;
    const u32 request_length =
      request_bytes == nullptr ? bytes : request_bytes[index];
    if (request_length == 0 || request_length > bytes ||
        request_length > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      return request_length > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
        ? -E2BIG : -EINVAL;
    }
  }
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
      local_iova, params.direct_local_mkey,
      request_bytes == nullptr ? bytes : request_bytes[index]);
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
  (void)owner_completion_timestamp_ns;
  (void)request_bytes;
  (void)priority;
  (void)nonblocking_enqueue;
  (void)prepared_matching;
  return -ENOTSUP;
#endif
}

// Publish a critical prefix and speculative suffix with one query-side ring
// transaction. The exclusive owner always places the prefix first. When the
// critical queue is drained and SQ credit remains, it posts
// [critical][CQ fence][tail][CQ fence] with one doorbell; otherwise the tail
// is rejected fail-soft without entering another queue. This removes the
// second query-side system fence/ring publication while preserving separate
// completion and validation lifetimes.
__device__ i32 direct_fetch_split_batch(
    const PersistentKernelParams& params,
    u32 memory_node,
    const u32* request_shards,
    const u64* remote_offsets,
    u32 request_count,
    u32 critical_request_count,
    u32 bytes,
    u32 lane,
    const u64* local_iova_offsets,
    i32* critical_completion,
    u64* critical_completion_timestamp_ns,
    i32* speculative_completion,
    u64* speculative_completion_timestamp_ns,
    const u32* request_bytes,
    u8 descriptor_flags = 0) {
#ifdef DVSTOR_HAVE_GPUNETIO
  const bool exact_snapshot_tail =
    (descriptor_flags & kDirectBatchFlagMandatoryFencedTail) != 0;
  const bool mixed_mandatory_tail =
    (descriptor_flags & kDirectBatchFlagMixedMandatoryFencedTail) != 0;
  const bool mandatory_fenced_tail =
    exact_snapshot_tail || mixed_mandatory_tail;
  if (critical_request_count == 0 ||
      critical_request_count > request_count ||
      (descriptor_flags & ~kDirectBatchKnownFlags) != 0 ||
      (exact_snapshot_tail && mixed_mandatory_tail) ||
      (exact_snapshot_tail &&
       request_count != 2u * critical_request_count) ||
      (mixed_mandatory_tail &&
       request_count < 2u * critical_request_count) ||
      (mandatory_fenced_tail && request_bytes != nullptr) ||
      request_shards == nullptr ||
      remote_offsets == nullptr ||
      local_iova_offsets == nullptr ||
      critical_completion == nullptr ||
      (!mandatory_fenced_tail && speculative_completion == nullptr) ||
      (mandatory_fenced_tail && speculative_completion != nullptr) ||
      params.direct_batch_queues == nullptr ||
      memory_node >= params.direct_region_count ||
      params.direct_qps == nullptr ||
      params.direct_qps_per_node == 0 ||
      params.direct_disabled == nullptr ||
      *reinterpret_cast<volatile u32*>(params.direct_disabled) != 0) {
    return -EINVAL;
  }
  if (mandatory_fenced_tail) {
    for (u32 index = 0; index < critical_request_count; ++index) {
      if (request_shards[index] !=
            request_shards[critical_request_count + index] ||
          remote_offsets[index] !=
            remote_offsets[critical_request_count + index] ||
          local_iova_offsets[index] + bytes !=
            local_iova_offsets[critical_request_count + index]) {
        return -EINVAL;
      }
    }
  }
  u32 critical_matching = 0;
  u32 speculative_matching = 0;
  for (u32 index = 0; index < request_count; ++index) {
    if (request_shards[index] != memory_node) continue;
    if (index < critical_request_count) {
      ++critical_matching;
    } else {
      ++speculative_matching;
    }
  }
  if (critical_matching == 0) {
    if (speculative_matching == 0) return 0;
    if (mandatory_fenced_tail) return -EINVAL;
    // This shard has no critical WQE train whose spare SQ credit can be
    // stolen, so publishing its tail would require a marginal doorbell.
    return direct_fetch_batch(
      params, memory_node,
      request_shards + critical_request_count,
      remote_offsets + critical_request_count,
      request_count - critical_request_count,
      params.graph_scratch, kPersistentGraphReadBytes, bytes, lane,
      local_iova_offsets + critical_request_count,
      speculative_completion, true, nullptr,
      speculative_completion_timestamp_ns,
      request_bytes == nullptr
        ? nullptr : request_bytes + critical_request_count,
      DirectBatchPriority::speculative, true);
  }
  if (speculative_matching == 0) {
    if (mandatory_fenced_tail) return -EINVAL;
    return direct_fetch_batch(
      params, memory_node, request_shards, remote_offsets,
      critical_request_count,
      params.graph_scratch, kPersistentGraphReadBytes, bytes, lane,
      local_iova_offsets, critical_completion, true, nullptr,
      critical_completion_timestamp_ns, request_bytes,
      DirectBatchPriority::critical, true);
  }

  const u32 qp_lane = lane % params.direct_qps_per_node;
  const u32 qp_index =
    qp_lane * params.direct_region_count + memory_node;
  if (qp_index >= params.direct_batch_queue_count ||
      params.direct_qps[qp_index] == nullptr) {
    return -EINVAL;
  }
  DirectOwnerProgress* watchdog_progress =
    params.direct_owner_progress == nullptr
      ? nullptr : params.direct_owner_progress + qp_index;
  atomicExch(critical_completion, -EINPROGRESS);
  if (speculative_completion != nullptr) {
    atomicExch(speculative_completion, -EINPROGRESS);
  }
  if (critical_completion_timestamp_ns != nullptr) {
    *critical_completion_timestamp_ns = 0;
  }
  if (speculative_completion_timestamp_ns != nullptr) {
    *speculative_completion_timestamp_ns = 0;
  }
  __threadfence();
  const DirectBatchDescriptor descriptor{
    .request_shards = request_shards,
    .remote_offsets = remote_offsets,
    .local_iova_offsets = local_iova_offsets,
    .request_bytes = request_bytes,
    .completion_status = critical_completion,
    .completion_timestamp_ns = critical_completion_timestamp_ns,
    .speculative_completion_status = speculative_completion,
    .speculative_completion_timestamp_ns =
      speculative_completion_timestamp_ns,
    .request_count = static_cast<u16>(request_count),
    .memory_node = static_cast<u16>(memory_node),
    .bytes = bytes,
    .priority = static_cast<u8>(DirectBatchPriority::critical),
    .flags = descriptor_flags,
    .critical_request_count =
      static_cast<u16>(critical_request_count),
  };
  if (watchdog_progress != nullptr) {
    record_owner_watchdog_counter(&watchdog_progress->announced);
  }
  bool pushed = device_ring_try_push(
    params.direct_batch_queues[qp_index], descriptor);
  while (!pushed && mandatory_fenced_tail) {
    if (*reinterpret_cast<const volatile u32*>(params.stop) != 0) {
      atomicExch(critical_completion, -ECANCELED);
      if (watchdog_progress != nullptr) {
        record_owner_watchdog_counter(&watchdog_progress->completed);
      }
      return -ECANCELED;
    }
    if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) != 0) {
      atomicExch(critical_completion, -EHOSTDOWN);
      if (watchdog_progress != nullptr) {
        record_owner_watchdog_counter(&watchdog_progress->completed);
      }
      return -EHOSTDOWN;
    }
    // This is correctness-critical work, so transient ring pressure is
    // bounded backpressure just like direct_fetch_batch(), not a reason to
    // fall back to two independently published batches.
    device_ring_relax(128);
    pushed = device_ring_try_push(
      params.direct_batch_queues[qp_index], descriptor);
  }
  if (!pushed) {
    atomicExch(critical_completion, -EAGAIN);
    if (speculative_completion != nullptr) {
      atomicExch(speculative_completion, -EAGAIN);
    }
    if (watchdog_progress != nullptr) {
      record_owner_watchdog_counter(&watchdog_progress->completed);
    }
    return -EAGAIN;
  }
  return -EINPROGRESS;
#else
  (void)params;
  (void)memory_node;
  (void)request_shards;
  (void)remote_offsets;
  (void)request_count;
  (void)critical_request_count;
  (void)bytes;
  (void)lane;
  (void)local_iova_offsets;
  (void)critical_completion;
  (void)critical_completion_timestamp_ns;
  (void)speculative_completion;
  (void)speculative_completion_timestamp_ns;
  (void)request_bytes;
  (void)descriptor_flags;
  return -ENOTSUP;
#endif
}

// Publish one exact-record snapshot train. The request SoA contains two
// equal-sized runs: complete records followed by second-header trailers.
// The owner reserves both runs as critical work, puts a hardware initiator
// fence on the first trailer READ, and signals only the final trailer (or one
// final dump WQE). Thus the caller observes one completion for
// [full snapshots -> fence -> validation snapshots] and never performs the
// old query-side wait/enqueue/wait round trip.
__device__ __forceinline__ i32 direct_fetch_fenced_snapshot_batch(
    const PersistentKernelParams& params,
    u32 memory_node,
    const u32* request_shards,
    const u64* remote_offsets,
    u32 snapshot_count,
    u32 bytes,
    u32 lane,
    const u64* local_iova_offsets,
    i32* completion,
    u64* completion_timestamp_ns = nullptr) {
  static_assert(2u * kPersistentMaxExact <=
                kPersistentMaxMergeCandidates);
  if (snapshot_count == 0 ||
      snapshot_count > kPersistentMaxExact) {
    return -EINVAL;
  }
  return direct_fetch_split_batch(
    params, memory_node, request_shards, remote_offsets,
    2u * snapshot_count, snapshot_count, bytes, lane,
    local_iova_offsets, completion, completion_timestamp_ns,
    nullptr, nullptr, nullptr,
    kDirectBatchFlagMandatoryFencedTail);
}

__device__ i32 wait_direct_batch(const PersistentKernelParams& params,
                                 i32* owner_completion) {
#ifdef DVSTOR_HAVE_GPUNETIO
  if (owner_completion == nullptr) return -EINVAL;
  for (;;) {
    const i32 status = *reinterpret_cast<volatile i32*>(owner_completion);
    if (status != -EINPROGRESS) return status;
    if (*reinterpret_cast<const volatile u32*>(params.stop) != 0) {
      return -ECANCELED;
    }
    if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) != 0) {
      return -EHOSTDOWN;
    }
    // This descriptor is already owned by a bounded owner queue.  Queueing
    // delay must not poison every query; the posting owner enforces the actual
    // CQ timeout and publishes a transport-wide failure when appropriate.
    device_ring_relax(128);
  }
#else
  (void)params;
  (void)owner_completion;
  return -ENOTSUP;
#endif
}

__device__ bool exact_record_visible(const PersistentKernelParams& params,
                                     const u8* record, u64 handle) {
  const u64 before = *reinterpret_cast<const u64*>(record);
  const u64 after = *reinterpret_cast<const u64*>(
    record + params.node_record_bytes);
  const u32 expected_incarnation = remote_incarnation(handle);
  const u32 stored_incarnation = *reinterpret_cast<const u32*>(
    record + params.node_incarnation_offset);
  return before == after &&
    (before & (kNodeLockMask | kNodeDeletedMask)) == 0 &&
    static_cast<u32>(before >> kNodeHeaderIncarnationShift) ==
      expected_incarnation &&
    stored_incarnation == expected_incarnation;
}

// A terminal-horizon exact prefetch has a one-round lifetime: it starts only
// after the next critical graph wave has been published and is consumed after
// that final authoritative commit.  Keep only scalar control in shared
// memory.  The request SoA, per-shard completion words, and cached records
// live in the otherwise-unused second half of this query slot's exact-record
// region, so the optimization adds neither a device allocation nor a
// query-slot-sized shared array.
struct TerminalExactCacheState {
  u32 attempted{};
  u32 active{};
  // Set before any cache record can be handed to the ordinary miss path.
  // That path reuses exact_records[0, K), so no later exact attempt may match
  // the retained metadata against those overwritten payload slots.
  u32 consumed{};
  u32 candidate_count{};
  u32 issued_records{};
  u32 arrived_records{};
  u32 promoted_records{};
  u32 queue_rejects{};
  u32 miss_count{};
  u64 wasted_bytes{};
};

struct TerminalExactCacheScratch {
  u32* request_shards{};
  u64* remote_offsets{};
  u64* local_iova_offsets{};
  i32* shard_status{};
  u8* record_base{};
  u8* metadata_base{};
  size_t metadata_bytes{};
  bool valid{};
};

__device__ __forceinline__ TerminalExactCacheScratch
terminal_exact_cache_scratch(const PersistentKernelParams& params,
                             const QueryDescriptor& descriptor) {
  TerminalExactCacheScratch scratch{};
  if (params.exact_records == nullptr ||
      params.node_record_stride == 0 ||
      params.exact_width < 2u * kPersistentMaxBeam) {
    return scratch;
  }
  scratch.record_base =
    params.exact_records +
    static_cast<size_t>(descriptor.query_slot) *
      params.exact_width * params.node_record_stride;
  scratch.metadata_base =
    scratch.record_base +
    static_cast<size_t>(kPersistentMaxBeam) *
      params.node_record_stride;
  scratch.metadata_bytes =
    static_cast<size_t>(params.exact_width - kPersistentMaxBeam) *
      params.node_record_stride;

  uintptr_t cursor = reinterpret_cast<uintptr_t>(scratch.metadata_base);
  const uintptr_t end = cursor + scratch.metadata_bytes;
  constexpr u32 terminal_snapshot_requests =
    2u * kPersistentMaxBeam;
  scratch.request_shards = reinterpret_cast<u32*>(cursor);
  cursor += static_cast<uintptr_t>(terminal_snapshot_requests) * sizeof(u32);
  cursor = (cursor + alignof(u64) - 1u) &
    ~static_cast<uintptr_t>(alignof(u64) - 1u);
  scratch.remote_offsets = reinterpret_cast<u64*>(cursor);
  cursor += static_cast<uintptr_t>(terminal_snapshot_requests) * sizeof(u64);
  scratch.local_iova_offsets = reinterpret_cast<u64*>(cursor);
  cursor += static_cast<uintptr_t>(terminal_snapshot_requests) * sizeof(u64);
  cursor = (cursor + alignof(i32) - 1u) &
    ~static_cast<uintptr_t>(alignof(i32) - 1u);
  scratch.shard_status = reinterpret_cast<i32*>(cursor);
  cursor += static_cast<uintptr_t>(kPersistentMaxShards) * sizeof(i32);
  scratch.valid = cursor <= end;
  return scratch;
}

// A fixed-record payload is immutable after one physical incarnation is
// published: upsert allocates a different incarnation rather than changing
// id/vector bytes in place. The terminal wave is a fenced
// [full record -> current header] train, so equality of its two headers and
// the stored incarnation is a complete exact snapshot obtained during this
// query. A later update linearizes after that snapshot and cannot mutate the
// retained physical incarnation.
__device__ __forceinline__ bool terminal_exact_cache_payload_valid(
    const PersistentKernelParams& params,
    const u8* record,
    u64 handle) {
  const u64 header = *reinterpret_cast<const u64*>(record);
  const u64 trailer = *reinterpret_cast<const u64*>(
    record + params.node_record_bytes);
  const u32 expected_incarnation = remote_incarnation(handle);
  const u32 stored_incarnation = *reinterpret_cast<const u32*>(
    record + params.node_incarnation_offset);
  return
    header == trailer &&
    (header & (kNodeLockMask | kNodeDeletedMask)) == 0 &&
    static_cast<u32>(header >> kNodeHeaderIncarnationShift) ==
      expected_incarnation &&
    stored_incarnation == expected_incarnation;
}

// Publish one optional fenced snapshot wave without waiting. This descriptor is
// deliberately enqueued only after the already-certified next graph core, so
// it cannot overtake the query's dependency. The query-side producer applies
// the same bounded critical backpressure as ordinary exact snapshots; any
// transport failure still degrades to the established exact fallback.
__device__ void begin_terminal_exact_cache_prefetch(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    const u64* beam_handles,
    u32 beam_count,
    TerminalExactCacheState& state) {
  const TerminalExactCacheScratch scratch =
    terminal_exact_cache_scratch(params, descriptor);
  if (threadIdx.x == 0) {
    state.attempted = 1;
    state.active = 0;
    state.consumed = 0;
    state.candidate_count =
      min(beam_count, static_cast<u32>(kPersistentMaxBeam));
    state.issued_records = 0;
    state.arrived_records = 0;
    state.promoted_records = 0;
    state.queue_rejects = 0;
    state.miss_count = 0;
    state.wasted_bytes = 0;
  }
  __syncthreads();
  if (!scratch.valid || state.candidate_count == 0 ||
      params.direct_batch_queues == nullptr) {
    return;
  }

  for (u32 index = threadIdx.x;
       index < 2u * state.candidate_count;
       index += blockDim.x) {
    scratch.request_shards[index] = UINT32_MAX;
    scratch.remote_offsets[index] = 0;
    scratch.local_iova_offsets[index] =
      index < state.candidate_count ? 0u : params.node_record_bytes;
  }
  for (u32 index = threadIdx.x; index < state.candidate_count;
       index += blockDim.x) {
    const u64 handle = beam_handles[index];
    u64 raw = 0;
    u64 graph_offset = 0;
    u32 shard = 0;
    if (!resolve_handle(params, handle, raw, shard, graph_offset) ||
        shard >= params.num_shards) {
      continue;
    }
    scratch.request_shards[index] = shard;
    scratch.remote_offsets[index] =
      remote_byte_offset(raw) + params.node_meta_offset;
    u8* destination =
      scratch.record_base +
      static_cast<size_t>(index) * params.node_record_stride;
    scratch.local_iova_offsets[index] =
      reinterpret_cast<u64>(destination) -
      params.direct_local_iova_base;
    const u32 trailer = state.candidate_count + index;
    scratch.request_shards[trailer] = shard;
    scratch.remote_offsets[trailer] = scratch.remote_offsets[index];
    scratch.local_iova_offsets[trailer] =
      reinterpret_cast<u64>(destination + params.node_record_bytes) -
      params.direct_local_iova_base;
  }
  for (u32 shard = threadIdx.x; shard < kPersistentMaxShards;
       shard += blockDim.x) {
    scratch.shard_status[shard] =
      shard < params.num_shards ? 0 : -EINVAL;
  }
  __syncthreads();

  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    const i32 issue_status = direct_fetch_fenced_snapshot_batch(
      params, shard, scratch.request_shards, scratch.remote_offsets,
      state.candidate_count, params.node_record_bytes,
      (descriptor.query_slot + shard) % params.direct_qps_per_node,
      scratch.local_iova_offsets, scratch.shard_status + shard,
      nullptr);
    // The owner may publish success before the enqueue helper returns.
    // Never overwrite that completion with a stale -EINPROGRESS return.
    if (issue_status != -EINPROGRESS) {
      scratch.shard_status[shard] = issue_status;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (u32 index = 0; index < state.candidate_count; ++index) {
      const u32 shard = scratch.request_shards[index];
      if (shard >= params.num_shards) continue;
      const i32 status =
        *reinterpret_cast<volatile i32*>(scratch.shard_status + shard);
      if (status == 0 || status == -EINPROGRESS) {
        ++state.issued_records;
      } else if (status == -EAGAIN) {
        ++state.queue_rejects;
      }
    }
    state.active = state.issued_records != 0 ? 1u : 0u;
  }
  __syncthreads();
}

// Always call this before exact scratch or the query slot can be reused.
// Cache transport failures are not query failures: the authoritative fenced
// exact read remains the correctness fallback.
__device__ void drain_terminal_exact_cache_prefetch(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    TerminalExactCacheState& state) {
  if (state.active == 0) return;
  const TerminalExactCacheScratch scratch =
    terminal_exact_cache_scratch(params, descriptor);
  if (!scratch.valid) {
    if (threadIdx.x == 0) state.active = 0;
    __syncthreads();
    return;
  }
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    const i32 status =
      *reinterpret_cast<volatile i32*>(scratch.shard_status + shard);
    if (status == -EINPROGRESS) {
      scratch.shard_status[shard] =
        wait_direct_batch(params, scratch.shard_status + shard);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    state.arrived_records = 0;
    for (u32 index = 0; index < state.candidate_count; ++index) {
      const u32 shard = scratch.request_shards[index];
      if (shard < params.num_shards &&
          scratch.shard_status[shard] == 0) {
        ++state.arrived_records;
      }
    }
    state.active = 0;
  }
  __syncthreads();
}

__device__ bool approximate_handles_batch(const PersistentKernelParams& params,
                                          const QueryDescriptor& descriptor,
                                          const f32* query_lut,
                                          u64* handles,
                                          u32 count,
                                          f32* distances,
                                          u64* total_dynamic_cycles,
                                          u32* total_dynamic_candidates,
                                          u32* total_dynamic_reads,
                                          u32* total_incarnation_rejects,
                                          u32* total_cache_hits,
                                          u32* total_batch_deduplicated,
                                          u32* total_cache_publish_successes,
                                          u32* total_cache_first_occupancies,
                                          u32* total_cache_publish_races,
                                          u32* total_lookup_probe_exhaustions,
                                          u32* total_publish_probe_exhaustions,
                                          u32* total_lookup_probes,
                                          u32* max_lookup_probes) {
  __shared__ i32 shard_status[kPersistentMaxShards];
  __shared__ u32 failed;
  __shared__ u32 call_dynamic_candidates;
  __shared__ u32 call_dynamic_reads;
  __shared__ u32 call_incarnation_rejects;
  __shared__ u32 call_cache_hits;
  __shared__ u32 call_batch_deduplicated;
  __shared__ u32 call_cache_publish_successes;
  __shared__ u32 call_cache_first_occupancies;
  __shared__ u32 call_cache_publish_races;
  __shared__ u32 call_lookup_probe_exhaustions;
  __shared__ u32 call_publish_probe_exhaustions;
  __shared__ u32 call_lookup_probes;
  __shared__ u32 call_max_lookup_probes;
  __shared__ u64 call_rdma_started_cycles;
  __shared__ u64 call_rdma_cycles;
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) * kPersistentMaxMergeCandidates;
  u32* request_shards = params.dynamic_code_request_shards + request_base;
  u64* request_offsets = params.dynamic_code_request_offsets + request_base;
  u64* request_local_iova_offsets =
    params.dynamic_code_request_local_iovas + request_base;
  if (threadIdx.x == 0) {
    failed = 0;
    call_dynamic_candidates = 0;
    call_dynamic_reads = 0;
    call_incarnation_rejects = 0;
    call_cache_hits = 0;
    call_batch_deduplicated = 0;
    call_cache_publish_successes = 0;
    call_cache_first_occupancies = 0;
    call_cache_publish_races = 0;
    call_lookup_probe_exhaustions = 0;
    call_publish_probe_exhaustions = 0;
    call_lookup_probes = 0;
    call_max_lookup_probes = 0;
    call_rdma_started_cycles = 0;
    call_rdma_cycles = 0;
  }
  __syncthreads();
  for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
    const u64 handle = handles[index];
    request_shards[index] = UINT32_MAX;
    request_offsets[index] = 0;
    request_local_iova_offsets[index] = 0;
    distances[index] = FLT_MAX;
    if (handle == kInvalidDeviceHandle) continue;
    u32 static_ordinal = 0;
    if (static_ordinal_from_raw(params, handle, static_ordinal)) {
      if (static_ordinal < params.num_nodes) {
        distances[index] = approximate_entry(
          params, query_lut,
          params.pq_codes +
            static_cast<size_t>(static_ordinal) * params.pq_code_bytes);
      }
      continue;
    }

    u64 raw = 0;
    u64 graph_offset = 0;
    u32 shard = 0;
    if (!resolve_handle(params, handle, raw, shard, graph_offset)) continue;
    // Dynamic PQ is authoritative in the storage record. Centroid-route seeds
    // and discovered neighbors therefore use the same one-sided RDMA path.
    if (params.dynamic_code_records == nullptr || shard >= params.num_shards) continue;
    atomicAdd(&call_dynamic_candidates, 1u);
    // Dynamic records use the same physical-slot ordinal on storage and GPU.
    // There is no hash table, collision chain, or capacity-dependent lookup.
    const DeviceShardRegion& region = params.shards[shard];
    const u64 node_offset = remote_byte_offset(raw);
    const bool arena_available =
      params.dynamic_code_arena_states != nullptr &&
      params.dynamic_code_arena_records != nullptr;
    u64 arena_slot = 0;
    if (arena_available && dynamic_code_arena_slot_from_offset(
          region, node_offset, params.dynamic_code_arena_capacity,
          arena_slot)) {
      auto* state = params.dynamic_code_arena_states + arena_slot;
      const u32 observed = dynamic_arena_state_load(state);
      atomicAdd(&call_lookup_probes, 1u);
      atomicMax(&call_max_lookup_probes, 1u);
      if (dynamic_code_arena_state_matches(
            observed, remote_incarnation(handle))) {
        const u8* resident = params.dynamic_code_arena_records +
          static_cast<size_t>(arena_slot) * params.pq_code_bytes;
        distances[index] = approximate_entry(params, query_lut, resident);
        // Keep every payload load before the second state sample. Together
        // with the publisher's release store this is the device-scope
        // seqlock read side; legacy relaxed atomics are intentionally avoided.
        cuda::atomic_thread_fence(
          cuda::memory_order_acquire, cuda::thread_scope_device);
        const u32 after = dynamic_arena_state_load(state);
        if (dynamic_code_arena_read_stable(
              observed, after, remote_incarnation(handle))) {
          atomicAdd(&call_cache_hits, 1u);
          continue;
        }
        // A replacement reserved or republished the physical slot while its
        // payload was being scored.  Never attribute those bytes to the old
        // handle; retry through the incarnation+checksum-validated storage
        // path in this same scoring call.
        distances[index] = FLT_MAX;
        atomicAdd(&call_incarnation_rejects, 1u);
      }
    }
    // The merge frontier can contain the same tagged handle through multiple
    // graph parents.  Deduplicate only inside this finite scoring call: one
    // stable incarnation-checked record is scattered to all identical
    // consumers below.  No value survives the call, so slot reuse needs no
    // cache invalidation protocol and ABA behavior is unchanged.
    u32 duplicate_of = UINT32_MAX;
    for (u32 prior = 0; prior < index; ++prior) {
      if (handles[prior] == handle) {
        duplicate_of = prior;
        break;
      }
    }
    if (duplicate_of != UINT32_MAX) {
      request_shards[index] = UINT32_MAX - 1u;
      request_offsets[index] = duplicate_of;
      atomicAdd(&call_batch_deduplicated, 1u);
      continue;
    }
    atomicAdd(&call_dynamic_reads, 1u);
    request_shards[index] = shard;
    request_offsets[index] = node_offset + params.shards[shard].dynamic_code_offset;
    u8* destination = params.dynamic_code_records +
      (request_base + index) * params.dynamic_code_record_bytes;
    request_local_iova_offsets[index] =
      reinterpret_cast<u64>(destination) - params.direct_local_iova_base;
  }
  for (u32 shard = threadIdx.x; shard < params.num_shards; shard += blockDim.x) {
    shard_status[shard] = 0;
  }
  __syncthreads();

  // The steady-state path is all cache hits.  Do not enter the shard fan-out,
  // CQ polling, publication, or duplicate-scatter machinery when there is no
  // authoritative read to issue.
  if (call_dynamic_reads == 0) {
    if (threadIdx.x == 0) {
      if (total_dynamic_candidates != nullptr) {
        *total_dynamic_candidates += call_dynamic_candidates;
      }
      if (total_cache_hits != nullptr) {
        *total_cache_hits += call_cache_hits;
      }
      if (total_lookup_probe_exhaustions != nullptr) {
        *total_lookup_probe_exhaustions += call_lookup_probe_exhaustions;
      }
      if (total_lookup_probes != nullptr) {
        *total_lookup_probes += call_lookup_probes;
      }
      if (max_lookup_probes != nullptr) {
        *max_lookup_probes = max(*max_lookup_probes, call_max_lookup_probes);
      }
    }
    __syncthreads();
    return true;
  }

  if (threadIdx.x == 0) call_rdma_started_cycles = clock64();
  __syncthreads();

  for (u32 shard = threadIdx.x; shard < params.num_shards; shard += blockDim.x) {
    i32* owner_completion = params.direct_batch_statuses == nullptr ? nullptr :
      params.direct_batch_statuses +
        static_cast<size_t>(descriptor.query_slot) * params.num_shards + shard;
    shard_status[shard] = direct_fetch_batch(
        params, shard, request_shards, request_offsets, count,
        params.dynamic_code_records +
          request_base * params.dynamic_code_record_bytes,
        params.dynamic_code_record_bytes, params.dynamic_code_record_bytes,
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
  if (threadIdx.x == 0) {
    call_rdma_cycles = clock64() - call_rdma_started_cycles;
  }
  __syncthreads();

  for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
    const u32 shard = request_shards[index];
    if (shard >= params.num_shards) continue;
    if (shard_status[shard] != 0) {
      atomicExch(&failed, 1u);
      continue;
    }
    const u8* code = params.dynamic_code_records +
      (request_base + index) * params.dynamic_code_record_bytes;
    const u32 dynamic_tag =
      vamana::dynamic_navigation_code::load_u32_le(code);
    const u8* payload = code + sizeof(u32);
    const u8* checksum = payload + params.pq_code_bytes;
    if (dynamic_code_tag_incarnation(dynamic_tag) ==
          remote_incarnation(handles[index]) &&
        vamana::dynamic_navigation_code::validate(
          dynamic_tag, payload, params.pq_code_bytes, checksum)) {
      distances[index] = approximate_entry(
        params, query_lut, payload);
    } else {
      atomicAdd(&call_incarnation_rejects, 1u);
    }
  }
  __syncthreads();
  // Serialize publication within this CTA. Globally, old-incarnation ->
  // BUSY|new-incarnation reserves the physical slot and payload -> fence ->
  // new-incarnation publishes it. Incarnations only increase, so a delayed
  // reader can never replace a newer payload (the slot-reuse ABA case).
  if (threadIdx.x == 0 && params.dynamic_code_arena_states != nullptr &&
      params.dynamic_code_arena_records != nullptr) {
    for (u32 index = 0; index < count; ++index) {
      const u32 shard = request_shards[index];
      if (shard >= params.num_shards || shard_status[shard] != 0 ||
          distances[index] == FLT_MAX) {
        continue;
      }
      const u64 handle = handles[index];
      const u8* source = params.dynamic_code_records +
        (request_base + index) * params.dynamic_code_record_bytes;
      const u32 source_tag = *reinterpret_cast<const u32*>(source);
      const u32 desired_incarnation = remote_incarnation(handle);
      if (dynamic_code_tag_incarnation(source_tag) !=
          desired_incarnation) {
        continue;
      }
      const u64 node_offset = remote_byte_offset(handle);
      const DeviceShardRegion& region = params.shards[shard];
      u64 arena_slot = 0;
      if (!dynamic_code_arena_slot_from_offset(
            region, node_offset, params.dynamic_code_arena_capacity,
            arena_slot)) {
        continue;
      }
      const u8 desired_extent =
        dynamic_code_tag_extent_class(source_tag);
      const u32 desired = make_dynamic_code_tag(
        desired_incarnation, desired_extent);
      auto* state = params.dynamic_code_arena_states + arena_slot;
      u32 observed = dynamic_arena_state_load(state);
      if (dynamic_code_arena_state_matches(
            observed, desired_incarnation)) {
        // Concurrent reads of one immutable PQ payload may observe successive
        // graph tags. Refine a defensive UNKNOWN state from the freshly read
        // storage tag, otherwise retain the larger class, all without
        // republishing the code. A later checksum-valid graph snapshot remains
        // the authority if the tag itself raced an adjacency publication.
        u32 promoted = observed;
        while (dynamic_code_arena_refined_unknown_extent_state(
                 observed, desired_incarnation, desired_extent, promoted) ||
               dynamic_code_arena_promoted_extent_state(
                 observed, desired_incarnation, desired_extent, promoted)) {
          const u32 prior = dynamic_arena_state_compare_exchange(
            state, observed, promoted);
          if (prior == observed) break;
          observed = prior;
        }
        ++call_cache_publish_races;
        continue;
      }
      if (!dynamic_code_arena_can_publish(
            observed, desired_incarnation) ||
          dynamic_arena_state_compare_exchange(
            state, observed,
            kPersistentDynamicCodeArenaBusy | desired) != observed) {
        ++call_cache_publish_races;
        continue;
      }
      u8* destination = params.dynamic_code_arena_records +
        static_cast<size_t>(arena_slot) * params.pq_code_bytes;
      for (u32 byte = 0; byte < params.pq_code_bytes; ++byte) {
        destination[byte] = source[sizeof(u32) + byte];
      }
      dynamic_arena_state_publish(state, desired);
      ++call_cache_publish_successes;
      if (dynamic_code_arena_first_occupancy(observed)) {
        ++call_cache_first_occupancies;
      }
    }
  }
  __syncthreads();
  for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
    if (request_shards[index] != UINT32_MAX - 1u) continue;
    const u64 source_index = request_offsets[index];
    if (source_index < index) distances[index] = distances[source_index];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (total_dynamic_cycles != nullptr) {
      *total_dynamic_cycles += call_rdma_cycles;
    }
    if (total_dynamic_candidates != nullptr) {
      *total_dynamic_candidates += call_dynamic_candidates;
    }
    if (total_dynamic_reads != nullptr) {
      *total_dynamic_reads += call_dynamic_reads;
    }
    if (total_incarnation_rejects != nullptr) {
      *total_incarnation_rejects += call_incarnation_rejects;
    }
    if (total_cache_hits != nullptr) {
      *total_cache_hits += call_cache_hits;
    }
    if (total_batch_deduplicated != nullptr) {
      *total_batch_deduplicated += call_batch_deduplicated;
    }
    if (total_cache_publish_successes != nullptr) {
      *total_cache_publish_successes += call_cache_publish_successes;
    }
    if (total_cache_first_occupancies != nullptr) {
      *total_cache_first_occupancies += call_cache_first_occupancies;
    }
    if (total_cache_publish_races != nullptr) {
      *total_cache_publish_races += call_cache_publish_races;
    }
    if (total_lookup_probe_exhaustions != nullptr) {
      *total_lookup_probe_exhaustions += call_lookup_probe_exhaustions;
    }
    if (total_publish_probe_exhaustions != nullptr) {
      *total_publish_probe_exhaustions += call_publish_probe_exhaustions;
    }
    if (total_lookup_probes != nullptr) {
      *total_lookup_probes += call_lookup_probes;
    }
    if (max_lookup_probes != nullptr) {
      *max_lookup_probes = max(*max_lookup_probes, call_max_lookup_probes);
    }
  }
  __syncthreads();
  return failed == 0;
}

__device__ bool exactify_into_beam(const PersistentKernelParams& params,
                                   const QueryDescriptor& descriptor,
                                   const f32* query, u64* candidate_handles,
                                   u32* candidate_ids, f32* candidate_distances,
                                   u32 candidate_count, u64* beam_handles,
                                   u32* beam_ids, f32* beam_distances,
                                   u8* beam_expanded, u32& beam_count,
                                   u32* exact_reads,
                                   u32* exact_snapshot_train_batches,
                                   u32* exact_snapshot_train_fallbacks,
                                   u32 beam_capacity, bool reset_beam,
                                   u64* merge_handles, u32* merge_ids,
                                   f32* merge_distances, u8* merge_expanded) {
  constexpr i32 kFencedSnapshotComplete = 1;
  constexpr i32 kNoSnapshotRequests = 2;
  __shared__ i32 shard_status[kPersistentMaxShards];
  __shared__ u32 transport_failed;
  if (candidate_count == 0) {
    if (reset_beam) {
      if (threadIdx.x == 0) beam_count = 0;
    } else {
      // Terminal-cache hits already carry exact distances. They still require
      // the authoritative exact ordering, but not an empty per-shard RDMA
      // protocol or a Beam->merge->Beam round trip.
      sort_candidates(
        beam_handles, beam_ids, beam_distances, beam_expanded, beam_count);
      if (threadIdx.x == 0) {
        beam_count = min(beam_count, beam_capacity);
      }
    }
    __syncthreads();
    return true;
  }
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) *
      kPersistentMaxMergeCandidates;
  u32* request_shards = params.dynamic_code_request_shards + request_base;
  u64* request_offsets = params.dynamic_code_request_offsets + request_base;
  u64* request_local_iova_offsets =
    params.dynamic_code_request_local_iovas + request_base;
  for (u32 index = threadIdx.x; index < candidate_count;
       index += blockDim.x) {
    request_shards[index] = UINT32_MAX;
    request_offsets[index] = 0;
    request_shards[candidate_count + index] = UINT32_MAX;
    request_offsets[candidate_count + index] = 0;
    request_local_iova_offsets[candidate_count + index] = 0;
    candidate_ids[index] = UINT32_MAX;
    candidate_distances[index] = FLT_MAX;
    const u64 handle = candidate_handles[index];
    u64 raw = 0;
    u64 graph_offset = 0;
    u32 shard = 0;
    if (!resolve_handle(params, handle, raw, shard, graph_offset)) continue;
    request_offsets[index] =
      remote_byte_offset(raw) + params.node_meta_offset;
    request_shards[index] = shard;
    const u8* destination =
      params.exact_records +
      (static_cast<size_t>(descriptor.query_slot) * params.exact_width +
       index) * params.node_record_stride;
    request_local_iova_offsets[index] =
      reinterpret_cast<u64>(destination) - params.direct_local_iova_base;
    const u32 trailer_index = candidate_count + index;
    request_shards[trailer_index] = shard;
    request_offsets[trailer_index] = request_offsets[index];
    request_local_iova_offsets[trailer_index] =
      reinterpret_cast<u64>(destination + params.node_record_bytes) -
      params.direct_local_iova_base;
  }
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    shard_status[shard] = 0;
  }
  __syncthreads();

  // One owner descriptor owns both snapshots. Every populated shard gets
  // exactly one success CQE after its fenced trailer run; request metadata
  // remains immutable until that final completion is observed.
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    bool has_request = false;
    for (u32 index = 0; index < candidate_count; ++index) {
      if (request_shards[index] == shard) {
        has_request = true;
        break;
      }
    }
    if (!has_request) {
      shard_status[shard] = kNoSnapshotRequests;
      continue;
    }
    if (exact_snapshot_train_batches != nullptr) {
      // "batches" is the attempt denominator: exactly one increment for each
      // populated shard that enters the mandatory train path. Every attempt
      // either reaches kFencedSnapshotComplete or increments fallbacks below.
      atomicAdd(exact_snapshot_train_batches, 1u);
    }
    i32* owner_completion =
      params.direct_batch_statuses == nullptr ? nullptr :
        params.direct_batch_statuses +
          static_cast<size_t>(descriptor.query_slot) * params.num_shards +
          shard;
    shard_status[shard] =
      params.direct_batch_queues == nullptr || owner_completion == nullptr
        ? -EAGAIN
        : direct_fetch_fenced_snapshot_batch(
            params, shard, request_shards, request_offsets, candidate_count,
            params.node_record_bytes,
            (descriptor.query_slot + shard) % params.direct_qps_per_node,
            request_local_iova_offsets, owner_completion);
  }
  __syncthreads();
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    if (shard_status[shard] != -EINPROGRESS) continue;
    i32* owner_completion =
      params.direct_batch_statuses +
      static_cast<size_t>(descriptor.query_slot) * params.num_shards + shard;
    shard_status[shard] = wait_direct_batch(params, owner_completion);
  }
  __syncthreads();
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    if (shard_status[shard] == 0) {
      shard_status[shard] = kFencedSnapshotComplete;
    }
  }
  __syncthreads();

  // A transport/capacity failure falls back per shard to the established
  // two-publication protocol. Successful trains never reread either
  // snapshot. Mark their trailer entries inactive so the common fallback
  // trailer pass is a no-op for those shards.
  for (u32 index = threadIdx.x; index < candidate_count;
       index += blockDim.x) {
    const u32 shard = request_shards[index];
    if (shard != UINT32_MAX &&
        shard_status[shard] == kFencedSnapshotComplete) {
      request_shards[candidate_count + index] = UINT32_MAX;
    }
  }
  __syncthreads();
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    if (shard_status[shard] == kFencedSnapshotComplete ||
        shard_status[shard] == kNoSnapshotRequests) {
      continue;
    }
    if (exact_snapshot_train_fallbacks != nullptr) {
      atomicAdd(exact_snapshot_train_fallbacks, 1u);
    }
    i32* owner_completion =
      params.direct_batch_statuses == nullptr ? nullptr :
        params.direct_batch_statuses +
          static_cast<size_t>(descriptor.query_slot) * params.num_shards +
          shard;
    shard_status[shard] = direct_fetch_batch(
      params, shard, request_shards, request_offsets, candidate_count,
      params.exact_records +
        static_cast<size_t>(descriptor.query_slot) *
          params.exact_width * params.node_record_stride,
      params.node_record_stride, params.node_record_bytes,
      (descriptor.query_slot + shard) % params.direct_qps_per_node,
      request_local_iova_offsets, owner_completion, true);
  }
  __syncthreads();
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    if (shard_status[shard] != -EINPROGRESS) continue;
    i32* owner_completion =
      params.direct_batch_statuses +
      static_cast<size_t>(descriptor.query_slot) * params.num_shards + shard;
    shard_status[shard] = wait_direct_batch(params, owner_completion);
  }
  __syncthreads();
  for (u32 index = threadIdx.x; index < candidate_count;
       index += blockDim.x) {
    const u32 shard = request_shards[index];
    if (shard != UINT32_MAX && shard_status[shard] != 0) {
      request_shards[candidate_count + index] = UINT32_MAX;
    }
  }
  __syncthreads();
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    if (shard_status[shard] != 0) continue;
    i32* owner_completion =
      params.direct_batch_statuses == nullptr ? nullptr :
        params.direct_batch_statuses +
          static_cast<size_t>(descriptor.query_slot) * params.num_shards +
          shard;
    shard_status[shard] = direct_fetch_batch(
      params, shard, request_shards + candidate_count,
      request_offsets + candidate_count, candidate_count,
      params.exact_records +
        static_cast<size_t>(descriptor.query_slot) *
          params.exact_width * params.node_record_stride +
        params.node_record_bytes,
      params.node_record_stride, sizeof(u64),
      (descriptor.query_slot + shard) % params.direct_qps_per_node,
      request_local_iova_offsets + candidate_count, owner_completion, true);
  }
  __syncthreads();
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    if (shard_status[shard] != -EINPROGRESS) continue;
    i32* owner_completion =
      params.direct_batch_statuses +
      static_cast<size_t>(descriptor.query_slot) * params.num_shards + shard;
    shard_status[shard] = wait_direct_batch(params, owner_completion);
  }
  __syncthreads();
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    if (shard_status[shard] == kFencedSnapshotComplete ||
        shard_status[shard] == kNoSnapshotRequests) {
      shard_status[shard] = 0;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    transport_failed = 0;
    for (u32 shard = 0; shard < params.num_shards; ++shard) {
      if (exact_snapshot_transport_failed(shard_status[shard])) {
        transport_failed = 1;
        break;
      }
    }
  }
  __syncthreads();
  // Do not rank or publish a subset when any populated shard exhausted the
  // legacy fallback. Visibility rejects are evaluated below only after every
  // transport completed and remain ordinary per-candidate filtering.
  if (transport_failed != 0) return false;

  for (u32 index = threadIdx.x; index < candidate_count;
       index += blockDim.x) {
    const u32 shard = request_shards[index];
    if (shard == UINT32_MAX || shard_status[shard] != 0) continue;
    const u8* record =
      params.exact_records +
      (static_cast<size_t>(descriptor.query_slot) * params.exact_width +
       index) * params.node_record_stride;
    if (exact_record_visible(params, record, candidate_handles[index])) {
      candidate_ids[index] =
        *reinterpret_cast<const u32*>(record + kNodeIdOffset);
      candidate_distances[index] = exact_storage_distance(
        params, query, record + params.node_vector_offset);
    }
    atomicAdd(exact_reads, 1u);
  }
  __syncthreads();
  const u32 existing_count = reset_beam ? 0 : beam_count;
  const u32 merge_count = existing_count + candidate_count;
  if (!reset_beam && merge_count <= beam_capacity) {
    // The terminal cache is the only append caller. Its hits already occupy
    // Beam and its strict misses are disjoint candidate storage, so append the
    // misses and run the same authoritative sorter in place. This preserves
    // the exact comparator and visibility filtering while avoiding two full
    // Beam copies for the common small-miss case.
    for (u32 index = threadIdx.x; index < candidate_count;
         index += blockDim.x) {
      const u32 destination = existing_count + index;
      beam_handles[destination] = candidate_handles[index];
      beam_ids[destination] = candidate_ids[index];
      beam_distances[destination] = candidate_distances[index];
      beam_expanded[destination] = 0;
    }
    __syncthreads();
    sort_candidates(
      beam_handles, beam_ids, beam_distances, beam_expanded, merge_count);
    if (threadIdx.x == 0) {
      u32 valid = 0;
      while (valid < merge_count &&
             beam_handles[valid] != kInvalidDeviceHandle &&
             isfinite(beam_distances[valid]) &&
             beam_distances[valid] != FLT_MAX) {
        ++valid;
      }
      beam_count = min(valid, beam_capacity);
    }
    __syncthreads();
    return true;
  }
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
    while (valid < merge_count &&
           merge_handles[valid] != kInvalidDeviceHandle &&
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
  return true;
}

// Commit a terminal-horizon cache without weakening exact visibility:
//
//   1. Match the final authoritative Beam to immutable cached incarnations.
//   2. Validate the fenced second header captured by the terminal wave.
//   3. Materialize valid hits into an exact Beam.
//   4. Pass every miss to the unchanged fenced-snapshot implementation and
//      merge it with those hits.
//
// A cache transport error, a reused slot, or an invalid early payload is just
// a miss.  A successfully observed final locked/deleted/stale header is an
// authoritative visibility rejection and must not be retried at a later
// linearization point.
__device__ bool exactify_into_beam_with_terminal_cache(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    const f32* query,
    u64* candidate_handles,
    u32* candidate_ids,
    f32* candidate_distances,
    u32 candidate_count,
    u64* beam_handles,
    u32* beam_ids,
    f32* beam_distances,
    u8* beam_expanded,
    u32& beam_count,
    u32* exact_reads,
    u32* exact_snapshot_train_batches,
    u32* exact_snapshot_train_fallbacks,
    u32 beam_capacity,
    bool reset_beam,
    u64* merge_handles,
    u32* merge_ids,
    f32* merge_distances,
    u8* merge_expanded,
    TerminalExactCacheState& cache) {
  // The production terminal call resets Beam. Keep compatibility callers on
  // the original implementation rather than creating a second append policy.
  if (cache.attempted == 0 || cache.issued_records == 0 ||
      cache.consumed != 0 || !reset_beam) {
    return exactify_into_beam(
      params, descriptor, query, candidate_handles, candidate_ids,
      candidate_distances, candidate_count, beam_handles, beam_ids,
      beam_distances, beam_expanded, beam_count, exact_reads,
      exact_snapshot_train_batches, exact_snapshot_train_fallbacks,
      beam_capacity, reset_beam, merge_handles, merge_ids,
      merge_distances, merge_expanded);
  }

  drain_terminal_exact_cache_prefetch(
    params, descriptor, cache);
  if (threadIdx.x == 0) cache.consumed = 1;
  __syncthreads();
  const TerminalExactCacheScratch scratch =
    terminal_exact_cache_scratch(params, descriptor);
  if (!scratch.valid || cache.arrived_records == 0) {
    return exactify_into_beam(
      params, descriptor, query, candidate_handles, candidate_ids,
      candidate_distances, candidate_count, beam_handles, beam_ids,
      beam_distances, beam_expanded, beam_count, exact_reads,
      exact_snapshot_train_batches, exact_snapshot_train_fallbacks,
      beam_capacity, reset_beam, merge_handles, merge_ids,
      merge_distances, merge_expanded);
  }

  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) *
      kPersistentMaxMergeCandidates;
  u32* request_shards =
    params.dynamic_code_request_shards + request_base;
  u64* request_offsets =
    params.dynamic_code_request_offsets + request_base;
  u64* request_local_iova_offsets =
    params.dynamic_code_request_local_iovas + request_base;
  constexpr u32 kTerminalMissSlotBit = 1u << 31;
  constexpr u32 kTerminalSlotMask = ~kTerminalMissSlotBit;
  constexpr u32 kTerminalCandidateMetadataBase =
    3u * kPersistentMaxBeam;
  static_assert(
    kTerminalCandidateMetadataBase + kPersistentMaxExact <=
    kPersistentMaxMergeCandidates);
  __shared__ u32 terminal_cache_used_slots[
    (kPersistentMaxBeam + 31u) / 32u];
  __shared__ u32 terminal_cache_mixed_misses;

  // candidate_ids temporarily stores the matched cache slot.  The key is the
  // exact remote fixed-record address; payload/header incarnation validation
  // below prevents physical-slot ABA from becoming a false hit.
  for (u32 index = threadIdx.x; index < candidate_count;
       index += blockDim.x) {
    candidate_ids[index] = UINT32_MAX;
    candidate_distances[index] = FLT_MAX;
    merge_expanded[index] = 2;
    request_shards[kTerminalCandidateMetadataBase + index] = UINT32_MAX;
    request_offsets[kTerminalCandidateMetadataBase + index] = 0;
    const u64 handle = candidate_handles[index];
    u64 raw = 0;
    u64 graph_offset = 0;
    u32 shard = 0;
    if (!resolve_handle(params, handle, raw, shard, graph_offset) ||
        shard >= params.num_shards) {
      // This is an authoritative invalid handle, not a transport miss.
      candidate_distances[index] = -1.0f;
      continue;
    }
    const u64 remote_offset =
      remote_byte_offset(raw) + params.node_meta_offset;
    request_shards[kTerminalCandidateMetadataBase + index] = shard;
    request_offsets[kTerminalCandidateMetadataBase + index] = remote_offset;
    u32 matched_cache_slot = UINT32_MAX;
    // Frontier turnover is normally positional near convergence. Test the
    // issue-time Beam rank first, preserving the full address/incarnation
    // checks. Only the small rank-drift subset pays the general search.
    if (index < cache.candidate_count) {
      const u32 cache_shard = scratch.request_shards[index];
      const u8* record =
        scratch.record_base +
        static_cast<size_t>(index) * params.node_record_stride;
      if (cache_shard == shard &&
          scratch.remote_offsets[index] == remote_offset &&
          scratch.shard_status[cache_shard] == 0 &&
          terminal_exact_cache_payload_valid(params, record, handle)) {
        matched_cache_slot = index;
      }
    }
    for (u32 cache_slot = 0;
         matched_cache_slot == UINT32_MAX &&
         cache_slot < cache.candidate_count; ++cache_slot) {
      if (cache_slot == index) continue;
      const u32 cache_shard = scratch.request_shards[cache_slot];
      if (cache_shard != shard ||
          scratch.remote_offsets[cache_slot] != remote_offset ||
          scratch.shard_status[cache_shard] != 0) {
        continue;
      }
      const u8* record =
        scratch.record_base +
        static_cast<size_t>(cache_slot) * params.node_record_stride;
      if (!terminal_exact_cache_payload_valid(
            params, record, handle)) {
        continue;
      }
      matched_cache_slot = cache_slot;
      break;
    }
    if (matched_cache_slot != UINT32_MAX) {
      candidate_ids[index] = matched_cache_slot;
    }
  }
  __syncthreads();

  // Claim every cache hit exactly once, then inject every pre-known miss into
  // a distinct unclaimed record slot. With F <= B == 128, unique H hit slots
  // leave B-H >= F-H slots, so this never needs another device allocation.
  // The defensive no-slot state remains a strict fallback.
  if (threadIdx.x == 0) {
    for (u32 word = 0;
         word < (kPersistentMaxBeam + 31u) / 32u; ++word) {
      terminal_cache_used_slots[word] = 0;
    }
    for (u32 index = 0; index < candidate_count; ++index) {
      const u32 slot = candidate_ids[index];
      if (slot >= cache.candidate_count ||
          slot >= kPersistentMaxBeam) {
        candidate_ids[index] = UINT32_MAX;
        continue;
      }
      const u32 bit = 1u << (slot & 31u);
      u32& used = terminal_cache_used_slots[slot >> 5u];
      if ((used & bit) != 0) {
        candidate_ids[index] = UINT32_MAX;
      } else {
        used |= bit;
      }
    }
    u32 miss_count = 0;
    u32 next_free = 0;
    for (u32 index = 0; index < candidate_count; ++index) {
      if (candidate_distances[index] < 0.0f) continue;
      if (candidate_ids[index] != UINT32_MAX) {
        merge_expanded[index] = 0;
        continue;
      }
      while (next_free < kPersistentMaxBeam &&
             (terminal_cache_used_slots[next_free >> 5u] &
              (1u << (next_free & 31u))) != 0) {
        ++next_free;
      }
      if (next_free == kPersistentMaxBeam) continue;
      terminal_cache_used_slots[next_free >> 5u] |=
        1u << (next_free & 31u);
      candidate_ids[index] = kTerminalMissSlotBit | next_free;
      merge_expanded[index] = 1;
      ++miss_count;
      ++next_free;
    }
    terminal_cache_mixed_misses = miss_count;

    // One immutable SoA is shared by every shard descriptor. Strict misses
    // are represented by full-record prefixes followed by address-identical
    // fenced header trailers. Cache hits already carry that exact train from
    // the terminal-horizon wave and require no second query-side round trip.
    u32 miss_index = 0;
    for (u32 index = 0; index < candidate_count; ++index) {
      const u8 origin = merge_expanded[index];
      if (origin > 1) continue;
      const u32 shard =
        request_shards[kTerminalCandidateMetadataBase + index];
      const u64 remote_offset =
        request_offsets[kTerminalCandidateMetadataBase + index];
      const u32 slot = candidate_ids[index] & kTerminalSlotMask;
      u8* record = scratch.record_base +
        static_cast<size_t>(slot) * params.node_record_stride;
      if (origin == 1) {
        const u32 prefix = miss_index++;
        const u32 trailer = terminal_cache_mixed_misses + prefix;
        request_shards[prefix] = shard;
        request_offsets[prefix] = remote_offset;
        request_local_iova_offsets[prefix] =
          reinterpret_cast<u64>(record) - params.direct_local_iova_base;
        request_shards[trailer] = shard;
        request_offsets[trailer] = remote_offset;
        request_local_iova_offsets[trailer] =
          reinterpret_cast<u64>(record + params.node_record_bytes) -
          params.direct_local_iova_base;
      }
    }
  }
  __syncthreads();

  const u32 mixed_request_count =
    2u * terminal_cache_mixed_misses;
  // Prefetch status is no longer needed after the mapping barrier. Reuse the
  // same completion words for one mixed final validation wave.
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    scratch.shard_status[shard] = 0;
    bool has_miss = false;
    for (u32 index = 0; index < terminal_cache_mixed_misses; ++index) {
      has_miss |= request_shards[index] == shard;
    }
    i32 issue_status = 0;
    if (has_miss) {
      if (exact_snapshot_train_batches != nullptr) {
        atomicAdd(exact_snapshot_train_batches, 1u);
      }
      issue_status = direct_fetch_split_batch(
        params, shard, request_shards, request_offsets, mixed_request_count,
        terminal_cache_mixed_misses, params.node_record_bytes,
        (descriptor.query_slot + shard) % params.direct_qps_per_node,
        request_local_iova_offsets, scratch.shard_status + shard, nullptr,
        nullptr, nullptr, nullptr,
        kDirectBatchFlagMixedMandatoryFencedTail);
    }
    if (issue_status != -EINPROGRESS) {
      scratch.shard_status[shard] = issue_status;
    }
  }
  __syncthreads();
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    if (scratch.shard_status[shard] == -EINPROGRESS) {
      scratch.shard_status[shard] =
        wait_direct_batch(params, scratch.shard_status + shard);
    }
  }
  __syncthreads();

  for (u32 index = threadIdx.x; index < candidate_count;
       index += blockDim.x) {
    const u8 origin = merge_expanded[index];
    if (origin > 1) continue;
    const u32 cache_slot = candidate_ids[index] & kTerminalSlotMask;
    const u32 shard =
      request_shards[kTerminalCandidateMetadataBase + index];
    if (shard >= params.num_shards ||
        scratch.shard_status[shard] != 0) {
      // Validation transport failure: preserve the ordinary exact fallback.
      candidate_ids[index] = UINT32_MAX;
      candidate_distances[index] = FLT_MAX;
      continue;
    }
    const u8* record =
      scratch.record_base +
      static_cast<size_t>(cache_slot) * params.node_record_stride;
    const bool visible = origin == 1
      ? exact_record_visible(params, record, candidate_handles[index])
      : terminal_exact_cache_payload_valid(
          params, record, candidate_handles[index]);
    if (!visible) {
      // The fenced trailer is the visibility linearization point. Negative
      // distances are private markers and never reach a sorter.
      candidate_ids[index] = UINT32_MAX;
      candidate_distances[index] = -1.0f;
      continue;
    }
    candidate_ids[index] =
      *reinterpret_cast<const u32*>(record + kNodeIdOffset);
    const f32 distance = exact_storage_distance(
      params, query, record + params.node_vector_offset);
    candidate_distances[index] =
      finite_f32_bits(distance) && distance != FLT_MAX
        ? distance : -1.0f;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    u32 hit_count = 0;
    u32 miss_count = 0;
    u32 cache_resolved = 0;
    u32 promoted_count = 0;
    for (u32 index = 0; index < candidate_count; ++index) {
      const f32 distance = candidate_distances[index];
      if (distance >= 0.0f && distance != FLT_MAX &&
          candidate_ids[index] != UINT32_MAX) {
        beam_handles[hit_count] = candidate_handles[index];
        beam_ids[hit_count] = candidate_ids[index];
        beam_distances[hit_count] = distance;
        beam_expanded[hit_count] = 0;
        ++hit_count;
        ++cache_resolved;
        if (merge_expanded[index] == 0) ++promoted_count;
      } else if (distance == FLT_MAX) {
        // Forward compaction is overlap-safe: destination never exceeds the
        // source index already consumed by this scalar stable scan.
        candidate_handles[miss_count++] = candidate_handles[index];
      } else {
        // A coherent final header rejected this candidate.
        // Invalid/unresolvable handles never issued an exact read, matching
        // the established exactify_into_beam telemetry contract.
        if (merge_expanded[index] <= 1) ++cache_resolved;
      }
    }
    beam_count = hit_count;
    cache.promoted_records = promoted_count;
    cache.miss_count = miss_count;
    cache.wasted_bytes =
      static_cast<u64>(
        cache.arrived_records > promoted_count
          ? cache.arrived_records - promoted_count : 0u) *
      params.node_record_bytes;
    if (exact_reads != nullptr) {
      *exact_reads += cache_resolved;
    }
  }
  __syncthreads();

  // With reset_beam=false the established implementation merges exact cache
  // hits with all strict misses, performs the one authoritative exact sort,
  // and preserves its transport/failure semantics unchanged.
  return exactify_into_beam(
    params, descriptor, query, candidate_handles, candidate_ids,
    candidate_distances, cache.miss_count, beam_handles, beam_ids,
    beam_distances, beam_expanded, beam_count, exact_reads,
    exact_snapshot_train_batches, exact_snapshot_train_fallbacks,
    beam_capacity, false, merge_handles, merge_ids,
    merge_distances, merge_expanded);
}

__device__ graph_record_validation::SnapshotState classify_graph_record(
    const PersistentKernelParams& params, const u8* record, u64 handle) {
  return graph_record_validation::classify_snapshot(
    record, params.graph_entry_bytes, params.graph_degree,
    params.graph_entry_capacity, remote_incarnation(handle));
}

__device__ graph_record_validation::SnapshotState
classify_short_graph_record(
    const PersistentKernelParams& params,
    const u8* record,
    u32 transferred_bytes,
    u64 handle) {
  return graph_record_validation::classify_zero_extended_snapshot(
    record, transferred_bytes, params.graph_entry_bytes,
    params.graph_degree, params.graph_entry_capacity,
    remote_incarnation(handle));
}

// Validate only descriptors backed by the dedicated frontier-request SoA.
// Pointer identity is the type tag, avoiding a larger queue descriptor and
// preserving the generic exact-vector/dynamic-code transport ABI. One owner
// warp lane validates one graph record after the final CQE and before the
// release publication of completion_status.
__device__ void validate_frontier_owner_batch(
    const PersistentKernelParams& params,
    const DirectBatchDescriptor& descriptor,
    u32 memory_node,
    u32 lane) {
  if (params.speculative_graph_request_shards == nullptr ||
      params.speculative_graph_request_offsets == nullptr ||
      params.speculative_graph_request_local_iovas == nullptr ||
      params.speculative_graph_request_handles == nullptr ||
      params.speculative_graph_validation_states == nullptr ||
      descriptor.request_shards == nullptr ||
      descriptor.remote_offsets == nullptr ||
      descriptor.local_iova_offsets == nullptr) {
    return;
  }
  const size_t capacity =
    static_cast<size_t>(params.query_slots) *
    kPersistentFrontierRobCapacity;
  const uintptr_t shards_begin = reinterpret_cast<uintptr_t>(
    params.speculative_graph_request_shards);
  const uintptr_t shards_end =
    shards_begin + capacity * sizeof(u32);
  const uintptr_t descriptor_shards =
    reinterpret_cast<uintptr_t>(descriptor.request_shards);
  if (descriptor_shards < shards_begin ||
      descriptor_shards >= shards_end ||
      ((descriptor_shards - shards_begin) % sizeof(u32)) != 0) {
    return;
  }
  const size_t request_base =
    (descriptor_shards - shards_begin) / sizeof(u32);
  if (request_base + descriptor.request_count > capacity ||
      descriptor.remote_offsets !=
        params.speculative_graph_request_offsets + request_base ||
      descriptor.local_iova_offsets !=
        params.speculative_graph_request_local_iovas + request_base ||
      (params.speculative_graph_request_bytes != nullptr &&
       descriptor.request_bytes !=
         params.speculative_graph_request_bytes + request_base)) {
    return;
  }

  for (u32 index = lane; index < descriptor.request_count; index += 32) {
    if (descriptor.request_shards[index] != memory_node) continue;
    const u32 transferred_bytes = descriptor.request_bytes == nullptr
      ? descriptor.bytes : descriptor.request_bytes[index];
    const u8* record = reinterpret_cast<const u8*>(
      params.direct_local_iova_base +
      descriptor.local_iova_offsets[index]);
    const u64 handle =
      params.speculative_graph_request_handles[request_base + index];
    u32 required_bytes = 0;
    const bool prefix_valid =
      graph_record_validation::required_live_extent_bytes(
        record, transferred_bytes, params.graph_degree,
        params.graph_entry_capacity, required_bytes);
    const bool partial = transferred_bytes < params.graph_entry_bytes;
    const bool extent_underhint =
      partial && prefix_valid && required_bytes > transferred_bytes;
    const graph_record_validation::SnapshotState snapshot =
      prefix_valid && required_bytes <= transferred_bytes
        ? (partial
            ? classify_short_graph_record(
                params, record, transferred_bytes, handle)
            : classify_graph_record(params, record, handle))
        : graph_record_validation::SnapshotState::invalid;
    const FrontierValidationState validation =
      extent_underhint
        ? FrontierValidationState::extent_underhint
        : snapshot == graph_record_validation::SnapshotState::valid
        ? FrontierValidationState::valid
        : snapshot ==
            graph_record_validation::SnapshotState::stale_incarnation
          ? FrontierValidationState::stale_incarnation
          : FrontierValidationState::invalid_snapshot;
    params.speculative_graph_validation_states[request_base + index] =
      static_cast<u8>(validation);
  }
  // Every owner lane publishes a disjoint validation byte. The owner warp
  // synchronizes before lane zero releases completion_status, but the release
  // flag belongs to a different thread; make each producer's global write
  // visible to the query CTA before that cross-thread publication.
  __threadfence();
}

__device__ __forceinline__ u8 load_graph_extent_class(
    const PersistentKernelParams& params, u32 static_ordinal) {
  if (params.graph_extent_class_words == nullptr ||
      static_ordinal >= params.num_nodes) {
    return graph_record_validation::kGraphExtentClassUnknown;
  }
  const u32 word = *reinterpret_cast<volatile const u32*>(
    params.graph_extent_class_words + static_ordinal / sizeof(u32));
  return graph_record_validation::packed_graph_extent_class(
    word, static_ordinal % sizeof(u32));
}

// Resolve the incarnation-scoped dynamic arena state for a graph handle.  PQ
// discovery publishes this word as payload -> fence -> tagged state.  An
// exact incarnation match is therefore sufficient to reuse its extent hint;
// BUSY, unknown, and recycled slots all fall back to the full graph record.
__device__ __forceinline__ bool dynamic_graph_extent_state(
    const PersistentKernelParams& params,
    u64 handle,
    u32*& state,
    u32& observed) {
  state = nullptr;
  observed = 0;
  if (params.dynamic_graph_extent_enabled == 0 ||
      params.dynamic_code_arena_states == nullptr ||
      remote_incarnation(handle) == 0) {
    return false;
  }
  u64 raw = 0;
  u64 graph_offset = 0;
  u32 shard = 0;
  if (!resolve_handle(params, handle, raw, shard, graph_offset) ||
      shard >= params.num_shards) {
    return false;
  }
  const DeviceShardRegion& region = params.shards[shard];
  u64 arena_slot = 0;
  if (!dynamic_code_arena_slot_from_offset(
        region, remote_byte_offset(raw),
        params.dynamic_code_arena_capacity, arena_slot)) {
    return false;
  }
  state = params.dynamic_code_arena_states + arena_slot;
  observed = dynamic_arena_state_load(state);
  return dynamic_code_arena_state_matches(
    observed, remote_incarnation(handle));
}

__device__ __forceinline__ u8 load_dynamic_graph_extent_class(
    const PersistentKernelParams& params, u64 handle) {
  u32* state = nullptr;
  u32 observed = 0;
  return dynamic_graph_extent_state(params, handle, state, observed)
    ? dynamic_code_tag_extent_class(observed)
    : kPersistentDynamicCodeArenaUnknownExtent;
}

// Once an under-hinted short read has been upgraded and the authoritative full
// snapshot validates, promote the packed device byte monotonically so later
// queries do not repeat the same dependent short->full pair. Seeing a stale
// class is harmless (one extra fallback), while a high-water class remains
// safe because full snapshot validation is always authoritative.
__device__ bool promote_graph_extent_class(
    const PersistentKernelParams& params,
    u64 handle,
    u32 required_bytes) {
  const u8 requested_class =
    graph_record_validation::graph_extent_class_for_required_bytes(
      required_bytes, params.graph_entry_capacity);
  if (requested_class ==
      graph_record_validation::kGraphExtentClassUnknown) {
    return false;
  }

  u32 static_ordinal = 0;
  if (static_ordinal_from_raw(params, handle, static_ordinal)) {
    if (params.graph_extent_class_words == nullptr ||
        static_ordinal >= params.num_nodes) {
      return false;
    }
    u32* const word =
      params.graph_extent_class_words + static_ordinal / sizeof(u32);
    const u32 byte_index = static_ordinal % sizeof(u32);
    u32 observed = *reinterpret_cast<volatile u32*>(word);
    while (true) {
      u32 desired = observed;
      if (!graph_record_validation::promoted_graph_extent_word(
            observed, byte_index, requested_class, desired)) {
        return false;
      }
      const u32 prior = atomicCAS(word, observed, desired);
      if (prior == observed) return true;
      observed = prior;
    }
  }

  u32* state = nullptr;
  u32 observed = 0;
  const u32 incarnation = remote_incarnation(handle);
  if (!dynamic_graph_extent_state(params, handle, state, observed)) {
    return false;
  }
  while (true) {
    u32 desired = observed;
    if (!dynamic_code_arena_promoted_extent_state(
          observed, incarnation, requested_class, desired)) {
      return false;
    }
    const u32 prior = dynamic_arena_state_compare_exchange(
      state, observed, desired);
    if (prior == observed) return true;
    observed = prior;
  }
}

enum class DynamicGraphExtentAdaptation : u8 {
  none = 0,
  refined_unknown = 1,
  demoted = 2,
};

// Adapt only dynamic hints and only after a checksum-valid snapshot. A
// defensive UNKNOWN state is refined to the exact observed class once; this
// planned full read is neither an underhint promotion nor a shrink demotion.
// Otherwise the two-class threshold and one-class guard avoid an atomic write
// for ordinary degree jitter. Static sidecar classes retain monotonic
// high-water semantics, while exact-incarnation CAS prevents an old query from
// modifying a slot that PQ publication has already reserved or recycled.
__device__ DynamicGraphExtentAdaptation adapt_dynamic_graph_extent_class(
    const PersistentKernelParams& params,
    u64 handle,
    u32 required_bytes) {
  if (remote_incarnation(handle) == 0) {
    return DynamicGraphExtentAdaptation::none;
  }
  const u8 observed_graph_class =
    graph_record_validation::graph_extent_class_for_required_bytes(
      required_bytes, params.graph_entry_capacity);
  if (observed_graph_class ==
      graph_record_validation::kGraphExtentClassUnknown) {
    return DynamicGraphExtentAdaptation::none;
  }
  u32* state = nullptr;
  u32 observed = 0;
  const u32 incarnation = remote_incarnation(handle);
  if (!dynamic_graph_extent_state(params, handle, state, observed)) {
    return DynamicGraphExtentAdaptation::none;
  }
  while (true) {
    u32 desired = observed;
    const bool refining_unknown =
      dynamic_code_tag_extent_class(observed) ==
        kPersistentDynamicCodeArenaUnknownExtent;
    const bool transition_available = refining_unknown
      ? dynamic_code_arena_refined_unknown_extent_state(
          observed, incarnation, observed_graph_class, desired)
      : dynamic_code_arena_guarded_demoted_extent_state(
          observed, incarnation, observed_graph_class, desired);
    if (!transition_available) {
      return DynamicGraphExtentAdaptation::none;
    }
    const u32 prior = dynamic_arena_state_compare_exchange(
      state, observed, desired);
    if (prior == observed) {
      return refining_unknown
        ? DynamicGraphExtentAdaptation::refined_unknown
        : DynamicGraphExtentAdaptation::demoted;
    }
    observed = prior;
  }
}

__device__ bool prepare_graph_record_in_scratch(
    const PersistentKernelParams& params,
    u64 handle,
    u32 query_slot,
    u32 scratch_slot,
    u32& acquired_slot,
    u32& request_shard,
    u64& request_offset,
    u64& request_local_iova,
    u32& request_bytes) {
  acquired_slot = UINT32_MAX;
  request_shard = UINT32_MAX;
  request_offset = 0;
  request_local_iova = 0;
  request_bytes = params.graph_entry_bytes;
  u64 raw = 0;
  u64 graph_offset = 0;
  u32 shard = 0;
  u32 static_ordinal = 0;
  const bool static_record =
    params.graph_extent_class_words != nullptr &&
    params.graph_request_bytes != nullptr &&
    static_ordinal_from_raw(params, handle, static_ordinal);
  if (!resolve_handle(params, handle, raw, shard, graph_offset)) return false;
  // Graph records are mutable and versioned by their storage checksum. Always
  // fetch the authoritative record so provisional backlinks and tombstones
  // need no query-side invalidation overlay.
  if (*reinterpret_cast<volatile u32*>(params.stop) != 0 ||
      params.graph_scratch == nullptr ||
      scratch_slot >= kPersistentGraphScratchSlots) {
    return false;
  }
  u8* destination = params.graph_scratch +
    (static_cast<size_t>(query_slot) * kPersistentGraphScratchSlots +
     scratch_slot) *
      kPersistentGraphReadBytes;
  acquired_slot = kGraphScratchBit | scratch_slot;
  request_shard = shard;
  request_offset = graph_offset;
  request_local_iova = reinterpret_cast<u64>(destination) -
    params.direct_local_iova_base;
  const bool header_neighbor =
    params.graph_read_policy ==
      static_cast<u32>(GraphReadPolicy::header_neighbor);
  // Both static sidecar and incarnation-scoped dynamic classes are hints:
  // validation below promotes any insufficient short read to an authoritative
  // full retry.  An unseen/BUSY/recycled dynamic slot returns the unknown class
  // and therefore preserves the legacy full-record path.
  if (header_neighbor && params.graph_request_bytes != nullptr) {
    request_bytes = graph_record_validation::kGraphRecordHeaderBytes;
  } else if (static_record && params.graph_extent_class_words != nullptr &&
      params.graph_request_bytes != nullptr &&
      (params.graph_entry_bytes & (sizeof(u64) - 1u)) == 0) {
    request_bytes = graph_record_validation::graph_extent_bytes_for_class(
      load_graph_extent_class(params, static_ordinal),
      params.graph_entry_bytes, params.graph_entry_capacity);
  } else if (!static_record && params.graph_request_bytes != nullptr &&
             (params.graph_entry_bytes & (sizeof(u64) - 1u)) == 0) {
    request_bytes = graph_record_validation::graph_extent_bytes_for_class(
      load_dynamic_graph_extent_class(params, handle),
      params.graph_entry_bytes, params.graph_entry_capacity);
  }
  return true;
}

__device__ bool prepare_graph_record(const PersistentKernelParams& params,
                                     u64 handle,
                                     u32 query_slot,
                                     u32 request_index,
                                     u32& acquired_slot,
                                     u32& request_shard,
                                     u64& request_offset,
                                     u64& request_local_iova,
                                     u32& request_bytes) {
  if (request_index >= kPersistentMaxPrefetch) return false;
  return prepare_graph_record_in_scratch(
    params, handle, query_slot, request_index, acquired_slot, request_shard,
    request_offset, request_local_iova, request_bytes);
}

__device__ u8* graph_record_pointer(const PersistentKernelParams& params,
                                    u32 query_slot, u32 acquired_slot) {
  if ((acquired_slot & kGraphScratchBit) != 0) {
    const u32 request_index = acquired_slot & ~kGraphScratchBit;
    if (request_index >= kPersistentGraphScratchSlots) return nullptr;
    return params.graph_scratch +
      (static_cast<size_t>(query_slot) * kPersistentGraphScratchSlots +
       request_index) *
        kPersistentGraphReadBytes;
  }
  return nullptr;
}

// Query-local state carried beside one authoritative graph fetch. A force-full
// retry is produced only after an asynchronous short prefix has already shown
// that the exact handle needs more bytes. The first admitted full attempt is
// therefore the full half of that one fallback; later checksum retries keep
// the underhint provenance for safe promotion but are not another fallback.
inline constexpr u32 kGraphReadLogical = 1u;
inline constexpr u32 kGraphReadStartedWithShortExtent = 2u;
inline constexpr u32 kGraphReadNeedsExtentFallback = 4u;
inline constexpr u32 kGraphReadExtentUnderhint = 8u;
inline constexpr u32 kGraphReadHeaderNeighborBody = 16u;

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr u32 prepare_graph_read_attempt_state(
    u32 full_record_bytes,
    bool force_full_underhint,
    u32& transfer_bytes) {
  if (force_full_underhint) {
    transfer_bytes = full_record_bytes;
    return kGraphReadLogical | kGraphReadNeedsExtentFallback |
      kGraphReadExtentUnderhint;
  }
  return kGraphReadLogical |
    (transfer_bytes < full_record_bytes
       ? kGraphReadStartedWithShortExtent : 0u);
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr u32 graph_read_state_after_fallback_admission(u32 state) {
  return state & ~kGraphReadNeedsExtentFallback;
}

struct GraphReadAdmissionAccounting {
  u32 fallback_reads{};
  u32 underhint_reads{};
  u32 retry_reads{};
};

// Classify one admitted physical read. Attempt zero may still be a physical
// retry when this helper was entered after an asynchronous short under-hint;
// later admitted attempts are ordinary checksum retries. Clearing the
// needs-fallback bit after admission makes the fallback count one-shot.
#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr GraphReadAdmissionAccounting
classify_graph_read_admission(
    u32 state, bool full_record_read, u32 batch_attempt) {
  const bool fallback = full_record_read &&
    (state & kGraphReadNeedsExtentFallback) != 0;
  return GraphReadAdmissionAccounting{
    .fallback_reads = fallback ? 1u : 0u,
    .underhint_reads =
      fallback && (state & kGraphReadExtentUnderhint) != 0 ? 1u : 0u,
    .retry_reads = batch_attempt != 0 ||
        (fallback && (state & kGraphReadStartedWithShortExtent) == 0)
      ? 1u : 0u,
  };
}

// Per-query physical accounting for dynamic graph traffic.  The structure is
// intended to live in shared memory and is passed by pointer through the RDMA
// helpers, avoiding six additional wide call arguments. Counts include every
// admitted physical snapshot attempt (including full retries), while fallback
// counts only the first admitted short->full upgrade.
struct DynamicGraphTelemetry {
  u64 read_bytes{};
  u32 short_reads{};
  u32 full_reads{};
  u32 fallback_reads{};
  u32 hint_promotions{};
  u32 hint_demotions{};
};

__device__ __forceinline__ bool dynamic_graph_telemetry_handle(u64 handle) {
  return handle != 0 && handle != kInvalidDeviceHandle &&
    remote_incarnation(handle) != 0;
}

__device__ __forceinline__ void add_dynamic_graph_read_telemetry(
    DynamicGraphTelemetry* telemetry,
    u32 short_reads,
    u32 full_reads,
    u64 read_bytes,
    u32 fallback_reads = 0) {
  if (telemetry == nullptr) return;
  if (short_reads != 0) atomicAdd(&telemetry->short_reads, short_reads);
  if (full_reads != 0) atomicAdd(&telemetry->full_reads, full_reads);
  if (read_bytes != 0) {
    atomicAdd(
      reinterpret_cast<unsigned long long*>(&telemetry->read_bytes),
      static_cast<unsigned long long>(read_bytes));
  }
  if (fallback_reads != 0) {
    atomicAdd(&telemetry->fallback_reads, fallback_reads);
  }
}

// Validate an asynchronously fetched frontier payload in the consuming query
// CTA. Publishing the CQ completion no longer depends on owner-side checksum
// work or a cross-thread system fence; the same snapshot/incarnation and
// Live-Extent rules are applied after the completion acquire.
__device__ FrontierValidationState validate_frontier_record_local(
    const PersistentKernelParams& params, u32 query_slot,
    const FrontierRobEntry& entry,
    DynamicGraphTelemetry* dynamic_telemetry = nullptr) {
  const u8* record = graph_record_pointer(
    params, query_slot,
    kGraphScratchBit | static_cast<u32>(entry.scratch_slot));
  if (record == nullptr || entry.transfer_bytes == 0) {
    return FrontierValidationState::invalid_snapshot;
  }
  u32 required_bytes = 0;
  const bool prefix_valid =
    graph_record_validation::required_live_extent_bytes(
      record, entry.transfer_bytes, params.graph_degree,
      params.graph_entry_capacity, required_bytes);
  if (!prefix_valid) {
    return FrontierValidationState::invalid_snapshot;
  }
  if (entry.transfer_bytes < params.graph_entry_bytes &&
      required_bytes > entry.transfer_bytes) {
    // This short prefix is not checksum-authoritative. Preserve only a
    // query-local force-full classification; the full retry below is the
    // first point allowed to repair the global extent hint.
    return FrontierValidationState::extent_underhint;
  }
  if (required_bytes > entry.transfer_bytes) {
    return FrontierValidationState::invalid_snapshot;
  }
  const graph_record_validation::SnapshotState snapshot =
    entry.transfer_bytes < params.graph_entry_bytes
      ? classify_short_graph_record(
          params, record, entry.transfer_bytes, entry.node_handle)
      : classify_graph_record(params, record, entry.node_handle);
  if (snapshot == graph_record_validation::SnapshotState::valid) {
    // Only a checksum-valid exact snapshot may move a dynamic hint down. The
    // helper applies hysteresis and rejects BUSY/recycled incarnations.
    if (adapt_dynamic_graph_extent_class(
          params, entry.node_handle, required_bytes) ==
          DynamicGraphExtentAdaptation::demoted &&
        dynamic_telemetry != nullptr) {
      atomicAdd(&dynamic_telemetry->hint_demotions, 1u);
    }
    return FrontierValidationState::valid;
  }
  return snapshot ==
      graph_record_validation::SnapshotState::stale_incarnation
    ? FrontierValidationState::stale_incarnation
    : FrontierValidationState::invalid_snapshot;
}

struct FrontierGraphBatchState {
  u32 active{};
  u32 fatal{};
  u32 rejected{};
  // Cumulative wait observation classifies a completely READY wave. Ordered
  // scoring uses READY-before-block directly and no longer needs a
  // most-recent wait flag.
  u32 finish_had_pending{};
  u64 issue_timestamp_ns[kPersistentMaxShards]{};
};

struct CoreFrontierTelemetry {
  u64* wait_cycles{};
  u64* completion_latency_ns{};
  u64* completion_groups{};
  u32* arrived{};
  u32* stale{};
  u32* ready_waves{};
  DynamicGraphTelemetry* dynamic_graph{};
};

// A split descriptor is visible to the query before the owner knows whether
// its speculative suffix fits the already-formed critical SQ train. Issue
// accounting therefore records the suffix optimistically. If the owner
// returns -EAGAIN without posting a tail WQE, the completion path publishes
// this exact one-shot correction so final graph-read telemetry remains
// physical rather than attempted.
struct TailAdmissionCorrection {
  u32 rejected_batches{};
  u32 rejected_reads{};
  u32 rejected_live_extent_reads{};
  u32 rejected_full_record_reads{};
  u64 rejected_bytes{};
};

struct TailFrontierTelemetry {
  u32* arrived{};
  u32* stale{};
  u64* wait_cycles{};
  u64* completion_latency_ns{};
  u64* completion_groups{};
  u64* wasted_bytes{};
  TailAdmissionCorrection* admission_correction{};
  DynamicGraphTelemetry* dynamic_graph{};
};

// A narrow core or tail frontier is owned by one warp. This avoids the
// general builder's CTA-wide setup barriers for both priority classes while
// preserving separate core/tail completion words inside one critical owner
// descriptor and one owner queue.
__device__ bool issue_narrow_frontier_graph_batch(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    FrontierRobEntry* rob,
    FrontierGraphBatchState& batch,
    u32 slot_begin,
    u32 slot_count,
    i32* batch_statuses,
    u64* batch_completion_timestamps_ns,
    DirectBatchPriority priority,
    u32* remote_batches,
    u32* total_remote_reads,
    u64* graph_read_bytes,
    u32* graph_live_extent_reads,
    u32* graph_full_record_reads,
    u32* classified_graph_reads,
    u64* classified_graph_bytes,
    u32* queue_rejects,
    u32* prefetch_reads = nullptr,
    u64* prefetch_bytes = nullptr,
    DynamicGraphTelemetry* dynamic_telemetry = nullptr) {
  constexpr u32 full_warp = 0xffffffffu;
  const u32 lane = threadIdx.x & 31u;
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) *
      kPersistentFrontierRobCapacity + slot_begin;
  u32* request_shards =
    params.speculative_graph_request_shards + request_base;
  u64* request_offsets =
    params.speculative_graph_request_offsets + request_base;
  u64* request_local_iovas =
    params.speculative_graph_request_local_iovas + request_base;
  u32* request_bytes = params.speculative_graph_request_bytes == nullptr
    ? nullptr : params.speculative_graph_request_bytes + request_base;
  u64* request_handles =
    params.speculative_graph_request_handles == nullptr
      ? nullptr
      : params.speculative_graph_request_handles + request_base;
  u8* validation_states =
    params.speculative_graph_validation_states == nullptr
      ? nullptr
      : params.speculative_graph_validation_states + request_base;
  i32* completion_statuses = batch_statuses +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  u64* completion_timestamps =
    batch_completion_timestamps_ns +
      static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  const bool live_extent_enabled = request_bytes != nullptr;

  if (threadIdx.x < 32) {
    // One warp exclusively owns this query-local issue wave.  Accumulate all
    // per-shard accounting in registers and publish it once from lane zero;
    // atomically updating the same shared counters for every populated shard
    // only serializes the enqueue path and provides no inter-CTA protection.
    u32 admitted_batches = 0;
    u32 admitted_reads = 0;
    u32 admitted_short_reads = 0;
    u32 admitted_full_reads = 0;
    u32 admitted_dynamic_short_reads = 0;
    u32 admitted_dynamic_full_reads = 0;
    u32 rejected_reads = 0;
    u32 fatal_batches = 0;
    u64 admitted_bytes = 0;
    u64 admitted_dynamic_bytes = 0;
    if (lane == 0) {
      batch.active = 0;
      batch.fatal = 0;
      batch.rejected = 0;
      batch.finish_had_pending = 0;
    }
    for (u32 shard = lane; shard < params.num_shards; shard += 32) {
      batch.issue_timestamp_ns[shard] = 0;
    }
    if (lane < slot_count) {
      request_shards[lane] = UINT32_MAX;
      request_offsets[lane] = 0;
      request_local_iovas[lane] = 0;
      if (request_bytes != nullptr) {
        request_bytes[lane] = params.graph_entry_bytes;
      }
      if (request_handles != nullptr) {
        request_handles[lane] = kInvalidDeviceHandle;
      }
      if (validation_states != nullptr) {
        validation_states[lane] =
          static_cast<u8>(FrontierValidationState::unknown);
      }
      FrontierRobEntry& entry = rob[slot_begin + lane];
      if (entry.state ==
          static_cast<u8>(FrontierRequestState::issued)) {
        u32 acquired_slot = UINT32_MAX;
        u32 shard = UINT32_MAX;
        u64 offset = 0;
        u64 local_iova = 0;
        u32 transfer_bytes = params.graph_entry_bytes;
        if (prepare_graph_record_in_scratch(
              params, entry.node_handle, descriptor.query_slot,
              slot_begin + lane,
              acquired_slot, shard, offset, local_iova,
              transfer_bytes)) {
          entry.scratch_slot = static_cast<u8>(slot_begin + lane);
          entry.transfer_bytes = transfer_bytes;
          request_shards[lane] = shard;
          request_offsets[lane] = offset;
          request_local_iovas[lane] = local_iova;
          if (request_bytes != nullptr) {
            request_bytes[lane] = transfer_bytes;
          }
          if (request_handles != nullptr) {
            request_handles[lane] = entry.node_handle;
          }
        } else {
          entry.state = static_cast<u8>(FrontierRequestState::stale);
          entry.validation = static_cast<u8>(
            FrontierValidationState::transport_rejected);
        }
      }
    }
    for (u32 shard = lane; shard < params.num_shards; shard += 32) {
      completion_statuses[shard] = 0;
      completion_timestamps[shard] = 0;
    }
    __syncwarp(full_warp);

    // One lane owns one request.  Form same-shard groups directly in the
    // warp instead of assigning one lane per physical shard and rescanning
    // the whole ROB prefix for every shard.  Only each group leader publishes
    // an owner descriptor; all accounting remains a single warp reduction.
    const u32 shard = lane < slot_count
      ? request_shards[lane] : UINT32_MAX;
    const bool valid_request =
      lane < slot_count && shard != UINT32_MAX;
    const u32 lane_bytes = valid_request
      ? (live_extent_enabled
          ? request_bytes[lane] : params.graph_entry_bytes)
      : 0u;
    const u32 valid_mask =
      __ballot_sync(full_warp, valid_request);
    const u32 short_mask = __ballot_sync(
      full_warp,
      valid_request && lane_bytes < params.graph_entry_bytes);
    const bool dynamic_request = valid_request &&
      dynamic_graph_telemetry_handle(rob[slot_begin + lane].node_handle);
    const u32 dynamic_mask =
      __ballot_sync(full_warp, dynamic_request);
    if (valid_request) {
      const u32 shard_group =
        __match_any_sync(valid_mask, shard);
      const u32 leader = __ffs(shard_group) - 1u;
      const u32 matching = __popc(shard_group);
      const u32 payload_bytes =
        __reduce_add_sync(shard_group, lane_bytes);
      const u32 short_reads =
        __popc(shard_group & short_mask);
      const u32 full_reads = matching - short_reads;
      const u32 dynamic_matching =
        __popc(shard_group & dynamic_mask);
      const u32 dynamic_short_reads =
        __popc(shard_group & dynamic_mask & short_mask);
      const u32 dynamic_full_reads =
        dynamic_matching - dynamic_short_reads;
      const u32 dynamic_payload_bytes = __reduce_add_sync(
        shard_group, dynamic_request ? lane_bytes : 0u);
      if (lane == leader) {
        const u64 issue_timestamp_ns = global_time_ns();
        const i32 status = direct_fetch_batch(
          params, shard, request_shards, request_offsets,
          slot_count, params.graph_scratch, kPersistentGraphReadBytes,
          params.graph_entry_bytes,
          (descriptor.query_slot + shard) % params.direct_qps_per_node,
          request_local_iovas, completion_statuses + shard, true, nullptr,
          completion_timestamps + shard,
          live_extent_enabled ? request_bytes : nullptr,
          priority, true, matching);
        const bool admitted =
          status == 0 || status == -EINPROGRESS;
        if (admitted) {
          batch.issue_timestamp_ns[shard] = issue_timestamp_ns;
          admitted_batches = 1;
          admitted_reads = matching;
          admitted_bytes = payload_bytes;
          admitted_short_reads = short_reads;
          admitted_full_reads = full_reads;
          admitted_dynamic_bytes = dynamic_payload_bytes;
          admitted_dynamic_short_reads = dynamic_short_reads;
          admitted_dynamic_full_reads = dynamic_full_reads;
        } else {
          // Publish an immediate producer-side failure into the same
          // completion word consumed by the lane-local state transition.
          completion_statuses[shard] = status;
          if (status == -EAGAIN ||
              priority == DirectBatchPriority::speculative) {
            rejected_reads = matching;
          } else {
            fatal_batches = 1;
          }
        }
      }
    }
    __syncwarp(full_warp);
    if (valid_request) {
      const i32 status = completion_statuses[shard];
      if (status != 0 && status != -EINPROGRESS) {
        rob[slot_begin + lane].transfer_bytes = 0;
        if (priority == DirectBatchPriority::speculative) {
          request_shards[lane] = UINT32_MAX;
        }
      }
    }
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      admitted_batches +=
        __shfl_down_sync(full_warp, admitted_batches, offset);
      admitted_reads +=
        __shfl_down_sync(full_warp, admitted_reads, offset);
      admitted_short_reads +=
        __shfl_down_sync(full_warp, admitted_short_reads, offset);
      admitted_full_reads +=
        __shfl_down_sync(full_warp, admitted_full_reads, offset);
      admitted_dynamic_short_reads += __shfl_down_sync(
        full_warp, admitted_dynamic_short_reads, offset);
      admitted_dynamic_full_reads += __shfl_down_sync(
        full_warp, admitted_dynamic_full_reads, offset);
      rejected_reads +=
        __shfl_down_sync(full_warp, rejected_reads, offset);
      fatal_batches +=
        __shfl_down_sync(full_warp, fatal_batches, offset);
      admitted_bytes +=
        __shfl_down_sync(full_warp, admitted_bytes, offset);
      admitted_dynamic_bytes += __shfl_down_sync(
        full_warp, admitted_dynamic_bytes, offset);
    }
    if (lane == 0) {
      batch.active = admitted_batches != 0 ? 1u : 0u;
      batch.rejected += rejected_reads;
      batch.fatal |= fatal_batches != 0 ? 1u : 0u;
      *remote_batches += admitted_batches;
      *total_remote_reads += admitted_reads;
      *classified_graph_reads += admitted_reads;
      *graph_read_bytes += admitted_bytes;
      *classified_graph_bytes += admitted_bytes;
      *graph_live_extent_reads += admitted_short_reads;
      *graph_full_record_reads += admitted_full_reads;
      add_dynamic_graph_read_telemetry(
        dynamic_telemetry, admitted_dynamic_short_reads,
        admitted_dynamic_full_reads, admitted_dynamic_bytes);
      *queue_rejects += rejected_reads;
      if (prefetch_reads != nullptr) {
        *prefetch_reads += admitted_reads;
      }
      if (prefetch_bytes != nullptr) {
        *prefetch_bytes += admitted_bytes;
      }
    }
    __syncwarp(full_warp);
    if (lane < slot_count) {
      FrontierRobEntry& entry = rob[slot_begin + lane];
      if (entry.state ==
          static_cast<u8>(FrontierRequestState::issued)) {
        const u32 shard = request_shards[lane];
        if (shard != UINT32_MAX &&
            (completion_statuses[shard] == 0 ||
             completion_statuses[shard] == -EINPROGRESS)) {
          entry.state =
            static_cast<u8>(FrontierRequestState::inflight);
        } else {
          entry.state = static_cast<u8>(FrontierRequestState::stale);
          entry.validation = static_cast<u8>(
            FrontierValidationState::transport_rejected);
        }
      }
    }
  }
  __syncthreads();
  return batch.fatal == 0;
}

// Build the complete issue frontier once and publish at most one descriptor
// per shard.  One warp lane owns one ROB request and `match_any` forms the
// per-shard groups.  This is deliberately request-centric: the former
// shard-centric implementation kept every request/telemetry accumulator live
// across direct_fetch_split_batch(), compiled at the architectural register
// limit, and repeatedly scanned the 32-slot ROB once per shard.  Group leaders
// now publish the descriptor while the other request lanes retain only their
// own metadata.  Query-local telemetry is reduced once by the warp; no atomics
// are needed.
//
// The owner still splits the descriptor into a strict-priority critical head
// and a speculative suffix.  This helper changes only GPU issue overhead, not
// request ordering, completion lifetime, or the Issue/Commit contract.
__device__ bool issue_split_frontier_graph_batch(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    FrontierRobEntry* rob,
    FrontierGraphBatchState& core_batch,
    FrontierGraphBatchState& tail_batch,
    u32 request_count,
    u32 core_slot_count,
    i32* core_batch_statuses,
    u64* core_batch_completion_timestamps_ns,
    i32* tail_batch_statuses,
    u64* tail_batch_completion_timestamps_ns,
    u32* remote_batches,
    u32* total_remote_reads,
    u64* graph_read_bytes,
    u32* graph_live_extent_reads,
    u32* graph_full_record_reads,
    u32* critical_graph_reads,
    u64* critical_graph_bytes,
    u32* speculative_graph_reads,
    u64* speculative_graph_bytes,
    u32* core_queue_rejects,
    u32* speculative_queue_rejects,
    u32* core_prefetch_reads,
    u64* core_prefetch_bytes,
    u32* tail_admitted,
    DynamicGraphTelemetry* dynamic_telemetry = nullptr) {
  constexpr u32 full_warp = 0xffffffffu;
  const u32 lane = threadIdx.x & 31u;
  // `request_count` is the admitted frontier width for this epoch, not the
  // compile-time ROB capacity.  Keeping the descriptor and all shard scans
  // bounded by the live prefix avoids repeatedly walking unused ROB slots.
  request_count = min(
    request_count, static_cast<u32>(kPersistentFrontierRobCapacity));
  core_slot_count = min(core_slot_count, request_count);
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) *
      kPersistentFrontierRobCapacity;
  u32* request_shards =
    params.speculative_graph_request_shards + request_base;
  u64* request_offsets =
    params.speculative_graph_request_offsets + request_base;
  u64* request_local_iovas =
    params.speculative_graph_request_local_iovas + request_base;
  u32* request_bytes = params.speculative_graph_request_bytes == nullptr
    ? nullptr : params.speculative_graph_request_bytes + request_base;
  u64* request_handles =
    params.speculative_graph_request_handles == nullptr
      ? nullptr : params.speculative_graph_request_handles + request_base;
  u8* validation_states =
    params.speculative_graph_validation_states == nullptr
      ? nullptr
      : params.speculative_graph_validation_states + request_base;
  i32* core_statuses = core_batch_statuses +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  u64* core_timestamps = core_batch_completion_timestamps_ns +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  i32* tail_statuses = tail_batch_statuses +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  u64* tail_timestamps = tail_batch_completion_timestamps_ns +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  const bool live_extent_enabled = request_bytes != nullptr;

  if (core_batch.active != 0 || tail_batch.active != 0 ||
      core_slot_count == 0 ||
      core_slot_count >= kPersistentFrontierRobCapacity ||
      core_batch_statuses == nullptr ||
      core_batch_completion_timestamps_ns == nullptr ||
      tail_batch_statuses == nullptr ||
      tail_batch_completion_timestamps_ns == nullptr) {
    if (threadIdx.x == 0) {
      core_batch.fatal = 1;
      tail_batch.fatal = 1;
    }
    __syncthreads();
    return false;
  }

  if (threadIdx.x < 32) {
    if (lane == 0) {
      core_batch.active = 0;
      core_batch.fatal = 0;
      core_batch.rejected = 0;
      core_batch.finish_had_pending = 0;
      tail_batch.active = 0;
      tail_batch.fatal = 0;
      tail_batch.rejected = 0;
      tail_batch.finish_had_pending = 0;
    }
    for (u32 shard = lane; shard < params.num_shards; shard += 32) {
      core_batch.issue_timestamp_ns[shard] = 0;
      tail_batch.issue_timestamp_ns[shard] = 0;
    }
    if (lane < request_count) {
      request_shards[lane] = UINT32_MAX;
      request_offsets[lane] = 0;
      request_local_iovas[lane] = 0;
      if (request_bytes != nullptr) {
        request_bytes[lane] = params.graph_entry_bytes;
      }
      if (request_handles != nullptr) {
        request_handles[lane] = kInvalidDeviceHandle;
      }
      if (validation_states != nullptr) {
        validation_states[lane] =
          static_cast<u8>(FrontierValidationState::unknown);
      }
    }
    for (u32 shard = lane; shard < params.num_shards; shard += 32) {
      core_statuses[shard] = 0;
      core_timestamps[shard] = 0;
      tail_statuses[shard] = 0;
      tail_timestamps[shard] = 0;
    }
    __syncwarp(full_warp);

    if (lane < request_count) {
      FrontierRobEntry& entry = rob[lane];
      if (entry.state ==
          static_cast<u8>(FrontierRequestState::issued)) {
        u32 acquired_slot = UINT32_MAX;
        u32 shard = UINT32_MAX;
        u64 offset = 0;
        u64 local_iova = 0;
        u32 transfer_bytes = params.graph_entry_bytes;
        if (prepare_graph_record_in_scratch(
              params, entry.node_handle, descriptor.query_slot, lane,
              acquired_slot, shard, offset, local_iova, transfer_bytes)) {
          entry.scratch_slot = static_cast<u8>(lane);
          entry.transfer_bytes = transfer_bytes;
          request_shards[lane] = shard;
          request_offsets[lane] = offset;
          request_local_iovas[lane] = local_iova;
          if (request_bytes != nullptr) {
            request_bytes[lane] = transfer_bytes;
          }
          if (request_handles != nullptr) {
            request_handles[lane] = entry.node_handle;
          }
        } else {
          entry.state = static_cast<u8>(FrontierRequestState::stale);
          entry.validation = static_cast<u8>(
            FrontierValidationState::transport_rejected);
        }
      }
    }
    __syncwarp(full_warp);

    const u32 shard = lane < request_count
      ? request_shards[lane] : UINT32_MAX;
    const bool valid_request =
      lane < request_count && shard != UINT32_MAX;
    const u32 lane_bytes = valid_request
      ? (live_extent_enabled
          ? request_bytes[lane] : params.graph_entry_bytes)
      : 0u;
    const u32 valid_mask =
      __ballot_sync(full_warp, valid_request);
    const u32 core_mask = __ballot_sync(
      full_warp, valid_request && lane < core_slot_count);
    const u32 tail_mask = valid_mask & ~core_mask;
    const u32 short_mask = __ballot_sync(
      full_warp,
      valid_request && lane_bytes < params.graph_entry_bytes);
    const bool dynamic_request = valid_request &&
      dynamic_graph_telemetry_handle(rob[lane].node_handle);
    const u32 dynamic_mask =
      __ballot_sync(full_warp, dynamic_request);

    u32 admitted_core_reads = 0;
    u32 admitted_tail_reads = 0;
    u32 admitted_core_bytes = 0;
    u32 admitted_tail_bytes = 0;
    u32 admitted_short_reads = 0;
    u32 admitted_full_reads = 0;
    u32 admitted_dynamic_short_reads = 0;
    u32 admitted_dynamic_full_reads = 0;
    u32 admitted_dynamic_bytes = 0;
    u32 admitted_core_batches = 0;
    u32 admitted_tail_batches = 0;
    u32 rejected_core_reads = 0;
    u32 rejected_tail_reads = 0;
    u32 fatal_core_batches = 0;

    if (valid_request) {
      const u32 shard_group =
        __match_any_sync(valid_mask, shard);
      const u32 leader = __ffs(shard_group) - 1u;
      const u32 core_matching = __popc(shard_group & core_mask);
      const u32 tail_matching = __popc(shard_group & tail_mask);
      const u32 core_payload_bytes = __reduce_add_sync(
        shard_group, lane < core_slot_count ? lane_bytes : 0u);
      const u32 tail_payload_bytes = __reduce_add_sync(
        shard_group, lane < core_slot_count ? 0u : lane_bytes);
      const u32 short_reads = __popc(shard_group & short_mask);
      const u32 full_reads =
        core_matching + tail_matching - short_reads;
      const u32 dynamic_matching =
        __popc(shard_group & dynamic_mask);
      const u32 dynamic_short_reads =
        __popc(shard_group & dynamic_mask & short_mask);
      const u32 dynamic_full_reads =
        dynamic_matching - dynamic_short_reads;
      const u32 dynamic_payload_bytes = __reduce_add_sync(
        shard_group, dynamic_request ? lane_bytes : 0u);

      if (lane == leader) {
        // Timestamp before publication: the exclusive owner may dequeue and
        // complete a tiny batch before this producer warp resumes. Sampling
        // after direct_fetch_split_batch() could therefore make completion
        // appear earlier than issue and underflow unsigned latency telemetry.
        const u64 issued_ns = global_time_ns();
        const i32 issue_status = direct_fetch_split_batch(
          params, shard, request_shards, request_offsets,
          request_count, core_slot_count, params.graph_entry_bytes,
          (descriptor.query_slot + shard) % params.direct_qps_per_node,
          request_local_iovas, core_statuses + shard,
          core_timestamps + shard, tail_statuses + shard,
          tail_timestamps + shard,
          live_extent_enabled ? request_bytes : nullptr);
        const bool admitted =
          issue_status == 0 || issue_status == -EINPROGRESS;
        if (admitted) {
          if (core_matching != 0) {
            core_batch.issue_timestamp_ns[shard] = issued_ns;
            admitted_core_reads = core_matching;
            admitted_core_bytes = core_payload_bytes;
            admitted_core_batches = 1;
          }
          if (tail_matching != 0) {
            tail_batch.issue_timestamp_ns[shard] = issued_ns;
            admitted_tail_reads = tail_matching;
            admitted_tail_bytes = tail_payload_bytes;
            admitted_tail_batches = 1;
          }
          admitted_short_reads = short_reads;
          admitted_full_reads = full_reads;
          admitted_dynamic_short_reads = dynamic_short_reads;
          admitted_dynamic_full_reads = dynamic_full_reads;
          admitted_dynamic_bytes = dynamic_payload_bytes;
        } else {
          if (core_matching != 0) {
            core_statuses[shard] = issue_status;
            if (issue_status == -EAGAIN) {
              rejected_core_reads = core_matching;
            } else {
              fatal_core_batches = 1;
            }
          }
          if (tail_matching != 0) {
            // Speculation is lossy by design. Convert every immediate
            // owner-side rejection to the same retryable stale state.
            tail_statuses[shard] = -EAGAIN;
            rejected_tail_reads = tail_matching;
          }
        }
      }
    }

    // The leader-only values above are query-local.  One warp reduction is
    // cheaper than atomically publishing once per populated shard and keeps
    // all 64-bit counters out of the direct-fetch call's live range.
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      admitted_core_reads += __shfl_down_sync(
        full_warp, admitted_core_reads, offset);
      admitted_tail_reads += __shfl_down_sync(
        full_warp, admitted_tail_reads, offset);
      admitted_core_bytes += __shfl_down_sync(
        full_warp, admitted_core_bytes, offset);
      admitted_tail_bytes += __shfl_down_sync(
        full_warp, admitted_tail_bytes, offset);
      admitted_short_reads += __shfl_down_sync(
        full_warp, admitted_short_reads, offset);
      admitted_full_reads += __shfl_down_sync(
        full_warp, admitted_full_reads, offset);
      admitted_dynamic_short_reads += __shfl_down_sync(
        full_warp, admitted_dynamic_short_reads, offset);
      admitted_dynamic_full_reads += __shfl_down_sync(
        full_warp, admitted_dynamic_full_reads, offset);
      admitted_dynamic_bytes += __shfl_down_sync(
        full_warp, admitted_dynamic_bytes, offset);
      admitted_core_batches += __shfl_down_sync(
        full_warp, admitted_core_batches, offset);
      admitted_tail_batches += __shfl_down_sync(
        full_warp, admitted_tail_batches, offset);
      rejected_core_reads += __shfl_down_sync(
        full_warp, rejected_core_reads, offset);
      rejected_tail_reads += __shfl_down_sync(
        full_warp, rejected_tail_reads, offset);
      fatal_core_batches += __shfl_down_sync(
        full_warp, fatal_core_batches, offset);
    }
    if (lane == 0) {
      const u32 admitted_reads =
        admitted_core_reads + admitted_tail_reads;
      const u64 admitted_bytes =
        static_cast<u64>(admitted_core_bytes) + admitted_tail_bytes;
      core_batch.active = admitted_core_batches != 0 ? 1u : 0u;
      tail_batch.active = admitted_tail_batches != 0 ? 1u : 0u;
      core_batch.rejected = rejected_core_reads;
      tail_batch.rejected = rejected_tail_reads;
      core_batch.fatal = fatal_core_batches != 0 ? 1u : 0u;
      *remote_batches +=
        admitted_core_batches + admitted_tail_batches;
      *total_remote_reads += admitted_reads;
      *graph_read_bytes += admitted_bytes;
      *graph_live_extent_reads += admitted_short_reads;
      *graph_full_record_reads += admitted_full_reads;
      add_dynamic_graph_read_telemetry(
        dynamic_telemetry, admitted_dynamic_short_reads,
        admitted_dynamic_full_reads, admitted_dynamic_bytes);
      *critical_graph_reads += admitted_core_reads;
      *critical_graph_bytes += admitted_core_bytes;
      *speculative_graph_reads += admitted_tail_reads;
      *speculative_graph_bytes += admitted_tail_bytes;
      *core_queue_rejects += rejected_core_reads;
      *speculative_queue_rejects += rejected_tail_reads;
      *core_prefetch_reads += admitted_core_reads;
      *core_prefetch_bytes += admitted_core_bytes;
      // This is the single physical admission point for a split tail.
      // Counting resident INFLIGHT slots in later core-only rounds would
      // charge the same descriptor repeatedly and bias queue-pressure
      // feedback against contraction.
      *tail_admitted += admitted_tail_reads;
    }
    __syncwarp(full_warp);

    if (lane < request_count) {
      FrontierRobEntry& entry = rob[lane];
      if (entry.state ==
          static_cast<u8>(FrontierRequestState::issued)) {
        const i32 status = shard == UINT32_MAX
          ? -EINVAL
          : (lane < core_slot_count
              ? core_statuses[shard] : tail_statuses[shard]);
        // An owner can dequeue the just-published split descriptor and reject
        // its suffix before this producer warp reaches the state transition.
        // A nonzero tail issue token proves that the descriptor itself was
        // admitted and its reads/bytes were counted optimistically. Preserve
        // INFLIGHT in that race so the terminal tail drain performs the one
        // complete admission correction. An immediate ring-push rejection
        // has no token and still takes the ordinary STALE path below.
        const bool admitted_tail_rejected_by_owner =
          lane >= core_slot_count && status == -EAGAIN &&
          shard != UINT32_MAX &&
          tail_batch.issue_timestamp_ns[shard] != 0;
        if (status == 0 || status == -EINPROGRESS ||
            admitted_tail_rejected_by_owner) {
          entry.state =
            static_cast<u8>(FrontierRequestState::inflight);
        } else {
          // Rejected requests did not consume network bytes and must not be
          // charged as speculative waste when their terminal slot is reclaimed.
          entry.transfer_bytes = 0;
          entry.state =
            static_cast<u8>(FrontierRequestState::stale);
          entry.validation = static_cast<u8>(
            FrontierValidationState::transport_rejected);
        }
      }
    }
  }
  __syncthreads();
  return core_batch.fatal == 0 && tail_batch.fatal == 0;
}

// Prepare and enqueue one shadow-frontier wave without waiting for CQ
// completion. Request metadata, completion words, and payload slots are
// disjoint from every critical/dynamic-code path and remain immutable until
// finish_frontier_graph_batch() drains the admitted descriptors.
__device__ bool issue_frontier_graph_batch(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    FrontierRobEntry* rob,
    FrontierGraphBatchState& batch,
    u32 slot_begin,
    u32 slot_count,
    i32* batch_statuses,
    u64* batch_completion_timestamps_ns,
    DirectBatchPriority priority,
    bool nonblocking_enqueue,
    u32* remote_batches,
    u32* total_remote_reads,
    u64* graph_read_bytes,
    u32* graph_live_extent_reads,
    u32* graph_full_record_reads,
    u32* speculative_graph_reads,
    u64* speculative_graph_bytes,
    u32* speculative_queue_rejects,
    u32* prefetch_reads = nullptr,
    u64* prefetch_bytes = nullptr,
    DynamicGraphTelemetry* dynamic_telemetry = nullptr) {
  // Completion/status/request storage is wave-owned. Reusing it while an
  // owner still holds the prior descriptor is an ABA corruption, not
  // backpressure. The query pipeline normally defers new tail admission; keep
  // this guard as a fail-closed lifetime invariant.
  if (batch.active != 0) {
    if (threadIdx.x == 0) batch.fatal = 1;
    __syncthreads();
    return false;
  }
  if (slot_begin > kPersistentFrontierRobCapacity ||
      slot_count > kPersistentFrontierRobCapacity - slot_begin ||
      batch_statuses == nullptr ||
      batch_completion_timestamps_ns == nullptr) {
    if (threadIdx.x == 0) batch.fatal = 1;
    __syncthreads();
    return false;
  }
  if (blockDim.x == kApproximateSortThreadsCompact &&
      slot_count <= kPersistentFrontierRobCapacity &&
      speculative_graph_reads != nullptr &&
      speculative_graph_bytes != nullptr &&
      remote_batches != nullptr && total_remote_reads != nullptr &&
      graph_read_bytes != nullptr &&
      graph_live_extent_reads != nullptr &&
      graph_full_record_reads != nullptr &&
      speculative_queue_rejects != nullptr) {
    return issue_narrow_frontier_graph_batch(
      params, descriptor, rob, batch, slot_begin, slot_count,
      batch_statuses, batch_completion_timestamps_ns, priority,
      remote_batches, total_remote_reads,
      graph_read_bytes, graph_live_extent_reads, graph_full_record_reads,
      speculative_graph_reads, speculative_graph_bytes,
      speculative_queue_rejects, prefetch_reads, prefetch_bytes,
      dynamic_telemetry);
  }
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) *
      kPersistentFrontierRobCapacity + slot_begin;
  u32* request_shards =
    params.speculative_graph_request_shards + request_base;
  u64* request_offsets =
    params.speculative_graph_request_offsets + request_base;
  u64* request_local_iovas =
    params.speculative_graph_request_local_iovas + request_base;
  u32* request_bytes = params.speculative_graph_request_bytes == nullptr
    ? nullptr : params.speculative_graph_request_bytes + request_base;
  u64* request_handles =
    params.speculative_graph_request_handles == nullptr
      ? nullptr
      : params.speculative_graph_request_handles + request_base;
  u8* validation_states =
    params.speculative_graph_validation_states == nullptr
      ? nullptr
      : params.speculative_graph_validation_states + request_base;
  i32* completion_statuses = batch_statuses +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  u64* completion_timestamps =
    batch_completion_timestamps_ns +
      static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  const bool live_extent_enabled = request_bytes != nullptr;

  if (threadIdx.x == 0) {
    batch.active = 0;
    batch.fatal = 0;
    batch.rejected = 0;
    batch.finish_had_pending = 0;
  }
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    batch.issue_timestamp_ns[shard] = 0;
  }
  __syncthreads();
  for (u32 local = threadIdx.x; local < slot_count;
       local += blockDim.x) {
    request_shards[local] = UINT32_MAX;
    request_offsets[local] = 0;
    request_local_iovas[local] = 0;
    if (request_bytes != nullptr) {
      request_bytes[local] = params.graph_entry_bytes;
    }
    if (request_handles != nullptr) {
      request_handles[local] = kInvalidDeviceHandle;
    }
    if (validation_states != nullptr) {
      validation_states[local] =
        static_cast<u8>(FrontierValidationState::unknown);
    }
    const u32 slot = slot_begin + local;
    FrontierRobEntry& entry = rob[slot];
    if (entry.state !=
        static_cast<u8>(FrontierRequestState::issued)) {
      continue;
    }
    u32 acquired_slot = UINT32_MAX;
    u32 shard = UINT32_MAX;
    u64 offset = 0;
    u64 local_iova = 0;
    u32 transfer_bytes = params.graph_entry_bytes;
    const u32 scratch_slot = slot;
    if (!prepare_graph_record_in_scratch(
          params, entry.node_handle, descriptor.query_slot, scratch_slot,
          acquired_slot, shard, offset, local_iova, transfer_bytes)) {
      entry.state = static_cast<u8>(FrontierRequestState::stale);
      entry.validation =
        static_cast<u8>(FrontierValidationState::transport_rejected);
      continue;
    }
    entry.scratch_slot = static_cast<u8>(scratch_slot);
    entry.transfer_bytes = transfer_bytes;
    request_shards[local] = shard;
    request_offsets[local] = offset;
    request_local_iovas[local] = local_iova;
    if (request_bytes != nullptr) request_bytes[local] = transfer_bytes;
    if (request_handles != nullptr) {
      request_handles[local] = entry.node_handle;
    }
  }
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    completion_statuses[shard] = 0;
    completion_timestamps[shard] = 0;
  }
  __syncthreads();

  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    u32 matching = 0;
    u32 payload_bytes = 0;
    u32 short_reads = 0;
    u32 full_reads = 0;
    u32 dynamic_short_reads = 0;
    u32 dynamic_full_reads = 0;
    u64 dynamic_payload_bytes = 0;
    for (u32 local = 0; local < slot_count; ++local) {
      if (request_shards[local] != shard) continue;
      ++matching;
      const u32 bytes = live_extent_enabled
        ? request_bytes[local] : params.graph_entry_bytes;
      payload_bytes += bytes;
      if (bytes < params.graph_entry_bytes) {
        ++short_reads;
      } else {
        ++full_reads;
      }
      if (dynamic_graph_telemetry_handle(
            rob[slot_begin + local].node_handle)) {
        dynamic_payload_bytes += bytes;
        if (bytes < params.graph_entry_bytes) {
          ++dynamic_short_reads;
        } else {
          ++dynamic_full_reads;
        }
      }
    }
    if (matching == 0) continue;
    const u64 issue_timestamp_ns = global_time_ns();
    const i32 status = direct_fetch_batch(
      params, shard, request_shards, request_offsets,
      slot_count, params.graph_scratch,
      kPersistentGraphReadBytes, params.graph_entry_bytes,
      (descriptor.query_slot + shard) % params.direct_qps_per_node,
      request_local_iovas, completion_statuses + shard, true, nullptr,
      completion_timestamps + shard,
      live_extent_enabled ? request_bytes : nullptr,
      priority, nonblocking_enqueue);
    const bool admitted = status == 0 || status == -EINPROGRESS;
    if (admitted) {
      batch.issue_timestamp_ns[shard] = issue_timestamp_ns;
      atomicExch(&batch.active, 1u);
      atomicAdd(remote_batches, 1u);
      atomicAdd(total_remote_reads, matching);
      atomicAdd(speculative_graph_reads, matching);
      if (prefetch_reads != nullptr) {
        atomicAdd(prefetch_reads, matching);
      }
      atomicAdd(
        reinterpret_cast<unsigned long long*>(graph_read_bytes),
        static_cast<unsigned long long>(payload_bytes));
      atomicAdd(
        reinterpret_cast<unsigned long long*>(speculative_graph_bytes),
        static_cast<unsigned long long>(payload_bytes));
      if (prefetch_bytes != nullptr) {
        atomicAdd(
          reinterpret_cast<unsigned long long*>(prefetch_bytes),
          static_cast<unsigned long long>(payload_bytes));
      }
      if (short_reads != 0) {
        atomicAdd(graph_live_extent_reads, short_reads);
      }
      if (full_reads != 0) {
        atomicAdd(graph_full_record_reads, full_reads);
      }
      add_dynamic_graph_read_telemetry(
        dynamic_telemetry, dynamic_short_reads, dynamic_full_reads,
        dynamic_payload_bytes);
    } else if (status == -EAGAIN) {
      atomicAdd(&batch.rejected, matching);
      atomicAdd(speculative_queue_rejects, matching);
      // This payload never entered an owner queue. Keep request-count
      // rejection telemetry, but do not later classify its bytes as fetched
      // speculative waste.
      for (u32 local = 0; local < slot_count; ++local) {
        if (request_shards[local] == shard) {
          rob[slot_begin + local].transfer_bytes = 0;
        }
      }
    } else {
      // This function is used for the speculative tail in the persistent
      // query path.  Treat any enqueue rejection as a stale probe; only the
      // critical owner path is allowed to make a query fail-stop.
      if (priority == DirectBatchPriority::speculative) {
        atomicAdd(&batch.rejected, matching);
        atomicAdd(speculative_queue_rejects, matching);
        for (u32 local = 0; local < slot_count; ++local) {
          if (request_shards[local] == shard) {
            request_shards[local] = UINT32_MAX;
            rob[slot_begin + local].transfer_bytes = 0;
          }
        }
      } else {
        atomicExch(&batch.fatal, 1u);
      }
    }
  }
  __syncthreads();

  for (u32 local = threadIdx.x; local < slot_count;
       local += blockDim.x) {
    const u32 slot = slot_begin + local;
    FrontierRobEntry& entry = rob[slot];
    if (entry.state !=
        static_cast<u8>(FrontierRequestState::issued)) {
      continue;
    }
    const u32 shard = request_shards[local];
    if (shard == UINT32_MAX) continue;
    const i32 status = completion_statuses[shard];
    if (status == 0 || status == -EINPROGRESS) {
      entry.state = static_cast<u8>(FrontierRequestState::inflight);
    } else {
      entry.state = static_cast<u8>(FrontierRequestState::stale);
      entry.validation =
        static_cast<u8>(FrontierValidationState::transport_rejected);
    }
  }
  __syncthreads();
  return batch.fatal == 0;
}

__device__ __forceinline__ bool finish_core_frontier_graph_batch(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    FrontierRobEntry* rob,
    FrontierGraphBatchState& batch,
    u32 slot_count,
    i32* batch_statuses,
    u64* batch_completion_timestamps_ns,
    u64* wait_cycles,
    u64* completion_latency_ns,
    u64* completion_groups,
    u32* arrived,
    u32* stale,
    u32* ready_waves,
    DynamicGraphTelemetry* dynamic_telemetry = nullptr) {
  // No descriptor was admitted for this epoch. Avoid rescanning stale
  // per-shard completion words and the core ROB prefix; in a stable wide
  // frontier this is the common case because retained tail records already
  // cover the next commit prefix.
  if (batch.active == 0) return batch.fatal == 0;
  constexpr u32 full_warp = 0xffffffffu;
  const u32 lane = threadIdx.x & 31u;
  i32* completion_statuses = batch_statuses +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  u64* completion_timestamps =
    batch_completion_timestamps_ns +
      static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) *
      kPersistentFrontierRobCapacity;
  const u32* request_shards =
    params.speculative_graph_request_shards + request_base;
  __shared__ u64 core_wait_started_cycles;
  __shared__ u32 core_finish_had_pending;
  __shared__ u32 core_finish_failed;

  if (threadIdx.x < 32) {
    if (lane == 0) {
      core_wait_started_cycles = clock64();
      core_finish_had_pending = 0;
      core_finish_failed = 0;
    }
    __syncwarp(full_warp);
    bool lane_pending = false;
    bool lane_failed = false;
    for (u32 shard = lane; shard < params.num_shards; shard += 32) {
      if (completion_statuses[shard] == -EINPROGRESS) {
        lane_pending = true;
        completion_statuses[shard] =
          wait_direct_batch(params, completion_statuses + shard);
      }
      if (completion_statuses[shard] != 0 &&
          completion_statuses[shard] != -EAGAIN) {
        lane_failed = true;
      }
      if (completion_statuses[shard] == 0 &&
          batch.issue_timestamp_ns[shard] != 0) {
        const u64 owner_completed_ns = completion_timestamps[shard];
        const u64 completed_ns =
          owner_completed_ns == 0 ? global_time_ns() : owner_completed_ns;
        atomicAdd(
          reinterpret_cast<unsigned long long*>(completion_latency_ns),
          static_cast<unsigned long long>(
            completed_ns - batch.issue_timestamp_ns[shard]));
        atomicAdd(
          reinterpret_cast<unsigned long long*>(completion_groups), 1ULL);
        batch.issue_timestamp_ns[shard] = 0;
      }
    }
    const u32 pending_mask =
      __ballot_sync(full_warp, lane_pending);
    const u32 failed_mask =
      __ballot_sync(full_warp, lane_failed);
    if (lane == 0) {
      core_finish_had_pending = pending_mask != 0 ? 1u : 0u;
      core_finish_failed = failed_mask != 0 ? 1u : 0u;
      batch.finish_had_pending |= core_finish_had_pending;
    }
    __syncwarp(full_warp);

    bool local_arrived = false;
    bool local_stale = false;
    if (lane < slot_count) {
      FrontierRobEntry& entry = rob[lane];
      if (entry.state ==
          static_cast<u8>(FrontierRequestState::inflight)) {
        const u32 shard = request_shards[lane];
        if (shard == UINT32_MAX || completion_statuses[shard] != 0) {
          entry.state = static_cast<u8>(FrontierRequestState::stale);
          entry.validation = static_cast<u8>(
            FrontierValidationState::transport_rejected);
          local_stale = true;
        } else {
          entry.state = static_cast<u8>(FrontierRequestState::arrived);
          local_arrived = true;
          const FrontierValidationState validation =
            validate_frontier_record_local(
              params, descriptor.query_slot, entry, dynamic_telemetry);
          if (validation == FrontierValidationState::valid) {
            entry.state =
              static_cast<u8>(FrontierRequestState::validated);
            entry.validation =
              static_cast<u8>(FrontierValidationState::valid);
          } else {
            entry.state = static_cast<u8>(FrontierRequestState::stale);
            entry.validation = static_cast<u8>(validation);
            local_stale = true;
          }
        }
      }
    }
    const u32 arrived_mask =
      __ballot_sync(full_warp, local_arrived);
    const u32 stale_mask =
      __ballot_sync(full_warp, local_stale);
    if (lane == 0) {
      *arrived += __popc(arrived_mask);
      *stale += __popc(stale_mask);
      if (core_finish_had_pending == 0) ++*ready_waves;
      *wait_cycles += clock64() - core_wait_started_cycles;
      batch.active = 0;
      batch.fatal |= core_finish_failed;
    }
  }
  __syncthreads();
  return batch.fatal == 0;
}

// Consume every shard group that is READY at one observation boundary.
// wait_if_none selects between two progress modes:
//
//   * true: wait for the first completion, then harvest the complete visible
//     CQ backlog;
//   * false: return zero immediately when the CQ is empty so the caller can
//     consume already READY query-private parent work during that network gap.
//
// The tri-state result is -1 for a fatal batch, zero for a nonblocking empty
// observation, and one after useful progress. A single-group consumer paid
// two full shard scans, a wide call ABI, and four CTA barriers even when the
// completed shard exposed only parents behind a lower-rank hole. Batched harvest
// changes only private ROB arrival/validation state; Beam publication remains
// outside this function and therefore frontier-consistent.
__device__ __noinline__ i32 finish_next_core_frontier_group(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    FrontierRobEntry* rob,
    FrontierGraphBatchState& batch,
    u32 slot_count,
    const CoreFrontierTelemetry& telemetry,
    bool wait_if_none = true) {
  constexpr u32 full_warp = 0xffffffffu;
  const u32 lane = threadIdx.x & 31u;
  i32* completion_statuses = params.core_batch_statuses +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  u64* completion_timestamps =
    params.core_batch_completion_timestamps_ns +
      static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) *
      kPersistentFrontierRobCapacity;
  const u32* request_shards =
    params.speculative_graph_request_shards + request_base;
  __shared__ u64 ready_shards;
  __shared__ u64 group_exposed_wait_cycles;

  if (threadIdx.x == 0) {
    ready_shards = 0;
    group_exposed_wait_cycles = 0;
  }
  __syncthreads();
  if (batch.active == 0) return batch.fatal == 0 ? 1 : -1;

  // One control lane observes at most 64 completion words. Other warps remain
  // quiescent until there is useful validation/decode work.
  if (threadIdx.x == 0) {
    u64 wait_started_cycles = 0;
    for (;;) {
      u64 observed_ready = 0;
      for (u32 shard = 0; shard < params.num_shards; ++shard) {
        if (batch.issue_timestamp_ns[shard] == 0) continue;
        const i32 status =
          *reinterpret_cast<volatile i32*>(completion_statuses + shard);
        if (status != -EINPROGRESS) {
          observed_ready |= u64{1} << shard;
        }
      }
      if (observed_ready != 0) {
        ready_shards = observed_ready;
        if (wait_started_cycles != 0) {
          group_exposed_wait_cycles =
            clock64() - wait_started_cycles;
        }
        break;
      }
      batch.finish_had_pending = 1;
      if (!wait_if_none) break;
      if (wait_started_cycles == 0) {
        wait_started_cycles = clock64();
      }
      if (*reinterpret_cast<const volatile u32*>(params.stop) != 0) {
        group_exposed_wait_cycles =
          clock64() - wait_started_cycles;
        break;
      }
      if (*reinterpret_cast<const volatile u32*>(
            params.direct_disabled) != 0) {
        group_exposed_wait_cycles =
          clock64() - wait_started_cycles;
        break;
      }
      device_ring_relax(128);
    }
    // A nonblocking empty observation is ordinary backpressure: preserve the
    // live batch so the caller can cover the gap with buffered score work.
    // Only a blocking wait can leave without a completion after stop/direct
    // disable and therefore turn the batch fatal.
    if (ready_shards == 0 && wait_if_none) {
      batch.fatal = 1;
      batch.active = 0;
    }
  }
  __syncthreads();
  if (ready_shards == 0) {
    if (!wait_if_none && batch.active != 0 && batch.fatal == 0) {
      return 0;
    }
    if (threadIdx.x == 0) {
      *telemetry.wait_cycles +=
        group_exposed_wait_cycles;
    }
    __syncthreads();
    return -1;
  }

  bool local_arrived = false;
  bool local_stale = false;
  if (threadIdx.x < 32 && lane < slot_count) {
    const u32 shard = request_shards[lane];
    const bool in_ready_group =
      shard < kPersistentMaxShards &&
      (ready_shards & (u64{1} << shard)) != 0;
    FrontierRobEntry& entry = rob[lane];
    if (in_ready_group && entry.state ==
        static_cast<u8>(FrontierRequestState::inflight)) {
      const i32 status = completion_statuses[shard];
      if (status != 0) {
        entry.state = static_cast<u8>(FrontierRequestState::stale);
        entry.validation = static_cast<u8>(
          FrontierValidationState::transport_rejected);
        local_stale = true;
      } else {
        entry.state = static_cast<u8>(FrontierRequestState::arrived);
        local_arrived = true;
        const FrontierValidationState validation =
          validate_frontier_record_local(
            params, descriptor.query_slot, entry,
            telemetry.dynamic_graph);
        if (validation == FrontierValidationState::valid) {
          entry.state =
            static_cast<u8>(FrontierRequestState::validated);
          entry.validation =
            static_cast<u8>(FrontierValidationState::valid);
        } else {
          entry.state = static_cast<u8>(FrontierRequestState::stale);
          entry.validation = static_cast<u8>(validation);
          local_stale = true;
        }
      }
    }
  }
  if (threadIdx.x < 32) {
    const u32 arrived_mask =
      __ballot_sync(full_warp, local_arrived);
    const u32 stale_mask =
      __ballot_sync(full_warp, local_stale);
    if (lane == 0) {
      *telemetry.arrived += __popc(arrived_mask);
      *telemetry.stale += __popc(stale_mask);
      for (u32 shard = 0; shard < params.num_shards; ++shard) {
        if ((ready_shards & (u64{1} << shard)) == 0) continue;
        const i32 status = completion_statuses[shard];
        if (status != 0 && status != -EAGAIN) {
          batch.fatal = 1;
        }
        const u64 issue_ns = batch.issue_timestamp_ns[shard];
        if (status == 0 && issue_ns != 0) {
          const u64 owner_completed_ns =
            completion_timestamps[shard];
          const u64 completed_ns =
            owner_completed_ns == 0
              ? global_time_ns() : owner_completed_ns;
          *telemetry.completion_latency_ns +=
            completed_ns - issue_ns;
          ++*telemetry.completion_groups;
        }
        batch.issue_timestamp_ns[shard] = 0;
      }
      bool remaining = false;
      for (u32 shard = 0; shard < params.num_shards; ++shard) {
        remaining |= batch.issue_timestamp_ns[shard] != 0;
      }
      batch.active = remaining ? 1u : 0u;
      if (batch.active == 0 && batch.finish_had_pending == 0) {
        ++*telemetry.ready_waves;
      }
      *telemetry.wait_cycles +=
        group_exposed_wait_cycles;
    }
  }
  __syncthreads();
  return batch.fatal == 0 ? 1 : -1;
}

// Finish a narrow speculative wave with one warp.  Tail waves use at most the
// 32-entry query-local ROB, so the generic CTA path's global atomics and
// repeated block barriers are pure overhead.  Lane i owns tail slot i and
// ballot/reduction publishes the complete wave telemetry once.
__device__ __forceinline__ bool finish_narrow_speculative_frontier_batch(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    FrontierRobEntry* rob,
    FrontierGraphBatchState& batch,
    u32 slot_begin,
    u32 slot_count,
    i32* batch_statuses,
    u64* batch_completion_timestamps_ns,
    bool wait_for_completion,
    u32* speculative_arrived,
    u32* speculative_stale,
    u64* speculative_wait_cycles,
    u64* speculative_completion_latency_ns,
    u64* speculative_completion_groups,
    u64* speculative_wasted_bytes,
    DynamicGraphTelemetry* dynamic_telemetry = nullptr) {
  if (batch.active == 0) return batch.fatal == 0;
  constexpr u32 full_warp = 0xffffffffu;
  const u32 lane = threadIdx.x & 31u;
  i32* completion_statuses = batch_statuses +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  u64* completion_timestamps =
    batch_completion_timestamps_ns +
      static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) *
      kPersistentFrontierRobCapacity + slot_begin;
  const u32* request_shards =
    params.speculative_graph_request_shards + request_base;
  __shared__ u64 tail_finish_started_cycles;
  __shared__ u32 tail_finish_pending;
  __shared__ u32 tail_finish_failed;

  if (threadIdx.x < 32) {
    if (lane == 0) {
      tail_finish_started_cycles = clock64();
      tail_finish_pending = 0;
      tail_finish_failed = 0;
    }
    __syncwarp(full_warp);
    bool lane_pending = false;
    u64 lane_completion_latency_ns = 0;
    u32 lane_completion_groups = 0;
    u32 lane_rejected_batches = 0;
    for (u32 shard = lane; shard < params.num_shards; shard += 32) {
      i32 status =
        *reinterpret_cast<volatile i32*>(completion_statuses + shard);
      if (status == -EINPROGRESS) {
        lane_pending = true;
        if (wait_for_completion) {
          status = wait_direct_batch(params, completion_statuses + shard);
        }
      }
      // This is the speculative tail.  Any owner-side completion error is
      // converted to a stale ROB entry below; the exact critical path will
      // retry the handle.  A speculative probe must not turn a recoverable
      // QP/transport condition into a query failure.
      if (status == 0 && batch.issue_timestamp_ns[shard] != 0) {
        const u64 owner_completed_ns = completion_timestamps[shard];
        const u64 completed_ns =
          owner_completed_ns == 0 ? global_time_ns() : owner_completed_ns;
        lane_completion_latency_ns +=
          completed_ns - batch.issue_timestamp_ns[shard];
        ++lane_completion_groups;
        batch.issue_timestamp_ns[shard] = 0;
      } else if (status == -EAGAIN &&
                 batch.issue_timestamp_ns[shard] != 0) {
        // The owner rejected this entire shard suffix before posting any WQE.
        // Clear its issue token so repeated nonblocking observations cannot
        // apply the physical-telemetry correction twice.
        ++lane_rejected_batches;
        batch.issue_timestamp_ns[shard] = 0;
      }
    }
    const u32 pending_mask =
      __ballot_sync(full_warp, lane_pending);
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      lane_completion_latency_ns += __shfl_down_sync(
        full_warp, lane_completion_latency_ns, offset);
      lane_completion_groups += __shfl_down_sync(
        full_warp, lane_completion_groups, offset);
    }
    if (lane == 0) {
      tail_finish_pending = pending_mask != 0 ? 1u : 0u;
      tail_finish_failed = 0;
      *speculative_completion_latency_ns +=
        lane_completion_latency_ns;
      *speculative_completion_groups += lane_completion_groups;
    }
  }
  __syncthreads();
  if (!wait_for_completion && tail_finish_pending != 0) {
    return batch.fatal == 0;
  }

  if (threadIdx.x < 32) {
    bool local_arrived = false;
    bool local_stale = false;
    u64 local_wasted_bytes = 0;
    if (lane < slot_count) {
      FrontierRobEntry& entry = rob[slot_begin + lane];
      if (entry.state ==
          static_cast<u8>(FrontierRequestState::inflight)) {
        const u32 shard = request_shards[lane];
        const i32 status = shard == UINT32_MAX
          ? -EINVAL : completion_statuses[shard];
        if (status != 0) {
          entry.state = static_cast<u8>(FrontierRequestState::stale);
          entry.validation = static_cast<u8>(
            FrontierValidationState::transport_rejected);
          local_stale = true;
        } else {
          entry.state = static_cast<u8>(FrontierRequestState::arrived);
          local_arrived = true;
          const FrontierValidationState validation =
            validate_frontier_record_local(
              params, descriptor.query_slot, entry, dynamic_telemetry);
          if (validation == FrontierValidationState::valid) {
            entry.state =
              static_cast<u8>(FrontierRequestState::validated);
            entry.validation =
              static_cast<u8>(FrontierValidationState::valid);
          } else {
            entry.state = static_cast<u8>(FrontierRequestState::stale);
            entry.validation = static_cast<u8>(validation);
            local_stale = true;
          }
        }
        if (local_stale) local_wasted_bytes = entry.transfer_bytes;
      }
    }
    const u32 arrived_mask =
      __ballot_sync(full_warp, local_arrived);
    const u32 stale_mask =
      __ballot_sync(full_warp, local_stale);
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      local_wasted_bytes += __shfl_down_sync(
        full_warp, local_wasted_bytes, offset);
    }
    if (lane == 0) {
      *speculative_arrived += __popc(arrived_mask);
      *speculative_stale += __popc(stale_mask);
      *speculative_wasted_bytes += local_wasted_bytes;
      *speculative_wait_cycles +=
        clock64() - tail_finish_started_cycles;
      batch.active = 0;
      batch.fatal |= tail_finish_failed;
    }
  }
  __syncthreads();
  return batch.fatal == 0;
}

// Query-tail specialization of the narrow completion path.  Every persistent
// query uses the complete 32-slot ROB address space and the dedicated tail
// completion arrays, so carrying slot bounds, array pointers, six counter
// pointers, and a runtime wait flag through the generic dispatcher only
// enlarges the call ABI and spills them before its fast-path predicates.
//
// WaitForCompletion is a compile-time property: the per-round observation is
// nonblocking, while failure/final cleanup is blocking.  Both instantiations
// retain the same completion accounting, immutable-snapshot validation, and
// lossy speculative-error semantics as the former narrow dispatcher path.
template <bool WaitForCompletion>
__device__ __noinline__ bool finish_query_tail_frontier_batch(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    FrontierRobEntry* rob,
    FrontierGraphBatchState& batch,
    const TailFrontierTelemetry& telemetry) {
  if (batch.active == 0) return batch.fatal == 0;
  constexpr u32 full_warp = 0xffffffffu;
  const u32 lane = threadIdx.x & 31u;
  i32* completion_statuses = params.tail_batch_statuses +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  u64* completion_timestamps =
    params.tail_batch_completion_timestamps_ns +
      static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) *
      kPersistentFrontierRobCapacity;
  const u32* request_shards =
    params.speculative_graph_request_shards + request_base;
  __shared__ u64 query_tail_finish_started_cycles;
  __shared__ u32 query_tail_finish_pending;

  if (threadIdx.x < 32) {
    if (lane == 0) {
      query_tail_finish_started_cycles = clock64();
      query_tail_finish_pending = 0;
    }
    __syncwarp(full_warp);
    bool lane_pending = false;
    u64 lane_completion_latency_ns = 0;
    u32 lane_completion_groups = 0;
    u32 lane_rejected_batches = 0;
    for (u32 shard = lane; shard < params.num_shards; shard += 32) {
      i32 status =
        *reinterpret_cast<volatile i32*>(completion_statuses + shard);
      if (status == -EINPROGRESS) {
        lane_pending = true;
        if constexpr (WaitForCompletion) {
          status = wait_direct_batch(params, completion_statuses + shard);
        }
      }
      if (status == 0 && batch.issue_timestamp_ns[shard] != 0) {
        const u64 owner_completed_ns = completion_timestamps[shard];
        const u64 completed_ns =
          owner_completed_ns == 0 ? global_time_ns() : owner_completed_ns;
        lane_completion_latency_ns +=
          completed_ns - batch.issue_timestamp_ns[shard];
        ++lane_completion_groups;
        batch.issue_timestamp_ns[shard] = 0;
      } else if (status == -EAGAIN &&
                 batch.issue_timestamp_ns[shard] != 0) {
        // Keep the token until the complete tail wave is terminal. A
        // nonblocking observation may see this rejected shard before another
        // shard completes; consuming it here would apply the optimistic
        // physical-read correction twice.
        ++lane_rejected_batches;
      }
    }
    const u32 pending_mask =
      __ballot_sync(full_warp, lane_pending);
    const bool terminal_wave =
      WaitForCompletion || pending_mask == 0;
    if (terminal_wave) {
      for (u32 shard = lane; shard < params.num_shards; shard += 32) {
        if (completion_statuses[shard] == -EAGAIN &&
            batch.issue_timestamp_ns[shard] != 0) {
          batch.issue_timestamp_ns[shard] = 0;
        }
      }
    }
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      lane_completion_latency_ns += __shfl_down_sync(
        full_warp, lane_completion_latency_ns, offset);
      lane_completion_groups += __shfl_down_sync(
        full_warp, lane_completion_groups, offset);
      lane_rejected_batches += __shfl_down_sync(
        full_warp, lane_rejected_batches, offset);
    }
    if (lane == 0) {
      query_tail_finish_pending = pending_mask != 0 ? 1u : 0u;
      *telemetry.completion_latency_ns += lane_completion_latency_ns;
      *telemetry.completion_groups += lane_completion_groups;
      if (terminal_wave) {
        telemetry.admission_correction->rejected_batches +=
          lane_rejected_batches;
      }
    }
  }
  __syncthreads();
  if constexpr (!WaitForCompletion) {
    if (query_tail_finish_pending != 0) {
      return batch.fatal == 0;
    }
  }

  if (threadIdx.x < 32) {
    bool local_arrived = false;
    bool local_stale = false;
    bool local_admission_rejected = false;
    u64 local_wasted_bytes = 0;
    u64 local_rejected_bytes = 0;
    u32 local_rejected_live_extent_reads = 0;
    u32 local_rejected_full_record_reads = 0;
    u64 local_rejected_dynamic_bytes = 0;
    u32 local_rejected_dynamic_short_reads = 0;
    u32 local_rejected_dynamic_full_reads = 0;
    FrontierRobEntry& entry = rob[lane];
    if (entry.state ==
        static_cast<u8>(FrontierRequestState::inflight)) {
      const u32 shard = request_shards[lane];
      const i32 status = shard == UINT32_MAX
        ? -EINVAL : completion_statuses[shard];
      if (status != 0) {
        local_admission_rejected = status == -EAGAIN;
        entry.state = static_cast<u8>(FrontierRequestState::stale);
        entry.validation = static_cast<u8>(
          FrontierValidationState::transport_rejected);
        local_stale = true;
      } else {
        entry.state = static_cast<u8>(FrontierRequestState::arrived);
        local_arrived = true;
        const FrontierValidationState validation =
          validate_frontier_record_local(
            params, descriptor.query_slot, entry,
            telemetry.dynamic_graph);
        if (validation == FrontierValidationState::valid) {
          entry.state =
            static_cast<u8>(FrontierRequestState::validated);
          entry.validation =
            static_cast<u8>(FrontierValidationState::valid);
        } else {
          entry.state = static_cast<u8>(FrontierRequestState::stale);
          entry.validation = static_cast<u8>(validation);
          local_stale = true;
        }
      }
      if (local_admission_rejected) {
        local_rejected_bytes = entry.transfer_bytes;
        local_rejected_live_extent_reads =
          entry.transfer_bytes < params.graph_entry_bytes ? 1u : 0u;
        local_rejected_full_record_reads =
          entry.transfer_bytes < params.graph_entry_bytes ? 0u : 1u;
        if (dynamic_graph_telemetry_handle(entry.node_handle)) {
          local_rejected_dynamic_bytes = entry.transfer_bytes;
          local_rejected_dynamic_short_reads =
            entry.transfer_bytes < params.graph_entry_bytes ? 1u : 0u;
          local_rejected_dynamic_full_reads =
            entry.transfer_bytes < params.graph_entry_bytes ? 0u : 1u;
        }
        // This payload never reached the NIC. Keep the terminal ROB state for
        // exact retry, but do not later charge it as speculative waste.
        entry.transfer_bytes = 0;
      } else if (local_stale) {
        local_wasted_bytes = entry.transfer_bytes;
      }
    }
    const u32 arrived_mask =
      __ballot_sync(full_warp, local_arrived);
    const u32 stale_mask =
      __ballot_sync(
        full_warp, local_stale && !local_admission_rejected);
    const u32 rejected_mask =
      __ballot_sync(full_warp, local_admission_rejected);
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      local_wasted_bytes += __shfl_down_sync(
        full_warp, local_wasted_bytes, offset);
      local_rejected_bytes += __shfl_down_sync(
        full_warp, local_rejected_bytes, offset);
      local_rejected_live_extent_reads += __shfl_down_sync(
        full_warp, local_rejected_live_extent_reads, offset);
      local_rejected_full_record_reads += __shfl_down_sync(
        full_warp, local_rejected_full_record_reads, offset);
      local_rejected_dynamic_bytes += __shfl_down_sync(
        full_warp, local_rejected_dynamic_bytes, offset);
      local_rejected_dynamic_short_reads += __shfl_down_sync(
        full_warp, local_rejected_dynamic_short_reads, offset);
      local_rejected_dynamic_full_reads += __shfl_down_sync(
        full_warp, local_rejected_dynamic_full_reads, offset);
    }
    if (lane == 0) {
      *telemetry.arrived += __popc(arrived_mask);
      *telemetry.stale += __popc(stale_mask);
      *telemetry.wasted_bytes += local_wasted_bytes;
      telemetry.admission_correction->rejected_reads +=
        __popc(rejected_mask);
      telemetry.admission_correction->rejected_bytes +=
        local_rejected_bytes;
      telemetry.admission_correction->rejected_live_extent_reads +=
        local_rejected_live_extent_reads;
      telemetry.admission_correction->rejected_full_record_reads +=
        local_rejected_full_record_reads;
      // Split admission initially counts the complete core+tail descriptor.
      // If the owner later rejects only the tail before posting any WQE, remove
      // its dynamic traffic here exactly once alongside the existing deferred
      // aggregate correction. This keeps the DynaExtent counters physical even
      // if callers use an older TailAdmissionCorrection consumer.
      if (telemetry.dynamic_graph != nullptr) {
        telemetry.dynamic_graph->read_bytes -= min(
          telemetry.dynamic_graph->read_bytes,
          local_rejected_dynamic_bytes);
        telemetry.dynamic_graph->short_reads -= min(
          telemetry.dynamic_graph->short_reads,
          local_rejected_dynamic_short_reads);
        telemetry.dynamic_graph->full_reads -= min(
          telemetry.dynamic_graph->full_reads,
          local_rejected_dynamic_full_reads);
      }
      *telemetry.wait_cycles +=
        clock64() - query_tail_finish_started_cycles;
      batch.active = 0;
    }
  }
  __syncthreads();
  return batch.fatal == 0;
}

// Drain an admitted speculative wave and validate each immutable local
// snapshot. Snapshot invalidity is speculative waste, not a transport failure:
// if the handle later becomes critical it will use the full authoritative
// retry path. A real owner/QP error remains fail-stop.
__device__ bool finish_frontier_graph_batch(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    FrontierRobEntry* rob,
    FrontierGraphBatchState& batch,
    u32 slot_begin,
    u32 slot_count,
    i32* batch_statuses,
    u64* batch_completion_timestamps_ns,
    bool wait_for_completion,
    u32* speculative_arrived,
    u32* speculative_stale,
    u64* speculative_wait_cycles,
    u64* speculative_completion_latency_ns,
    u64* speculative_completion_groups,
    u64* speculative_wasted_bytes,
    u32* classified_arrived = nullptr,
    u32* classified_stale = nullptr,
    u64* classified_wasted_bytes = nullptr,
    u32* classified_ready_waves = nullptr,
    DynamicGraphTelemetry* dynamic_telemetry = nullptr) {
  if (batch.active == 0) return batch.fatal == 0;
  if (slot_begin > kPersistentFrontierRobCapacity ||
      slot_count > kPersistentFrontierRobCapacity - slot_begin ||
      batch_statuses == nullptr ||
      batch_completion_timestamps_ns == nullptr) {
    if (threadIdx.x == 0) batch.fatal = 1;
    __syncthreads();
    return false;
  }
  if (blockDim.x == kApproximateSortThreadsCompact &&
      slot_count <= kPersistentFrontierRobCapacity &&
      speculative_arrived != nullptr &&
      speculative_stale != nullptr &&
      speculative_wait_cycles != nullptr &&
      speculative_completion_latency_ns != nullptr &&
      speculative_completion_groups != nullptr &&
      speculative_wasted_bytes != nullptr &&
      classified_arrived == nullptr && classified_stale == nullptr &&
      classified_wasted_bytes == nullptr &&
      classified_ready_waves == nullptr) {
    return finish_narrow_speculative_frontier_batch(
      params, descriptor, rob, batch, slot_begin, slot_count,
      batch_statuses, batch_completion_timestamps_ns,
      wait_for_completion, speculative_arrived, speculative_stale,
      speculative_wait_cycles, speculative_completion_latency_ns,
      speculative_completion_groups, speculative_wasted_bytes,
      dynamic_telemetry);
  }
  if (blockDim.x == kApproximateSortThreadsCompact &&
      wait_for_completion && slot_begin == 0 &&
      slot_count <= kPersistentScoreChunk &&
      speculative_arrived == nullptr && speculative_stale == nullptr &&
      speculative_wasted_bytes == nullptr &&
      classified_arrived != nullptr && classified_stale != nullptr &&
      classified_ready_waves != nullptr &&
      speculative_wait_cycles != nullptr &&
      speculative_completion_latency_ns != nullptr &&
      speculative_completion_groups != nullptr) {
    return finish_core_frontier_graph_batch(
      params, descriptor, rob, batch, slot_count, batch_statuses,
      batch_completion_timestamps_ns, speculative_wait_cycles,
      speculative_completion_latency_ns, speculative_completion_groups,
      classified_arrived, classified_stale, classified_ready_waves,
      dynamic_telemetry);
  }
  i32* completion_statuses = batch_statuses +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  u64* completion_timestamps =
    batch_completion_timestamps_ns +
      static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  __shared__ u64 speculative_wait_started_cycles;
  __shared__ u32 speculative_finish_failed;
  __shared__ u32 speculative_finish_pending;
  if (threadIdx.x == 0) {
    speculative_wait_started_cycles = clock64();
    speculative_finish_failed = 0;
    speculative_finish_pending = 0;
  }
  __syncthreads();
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    if (completion_statuses[shard] == -EINPROGRESS) {
      atomicExch(&speculative_finish_pending, 1u);
      if (wait_for_completion) {
        completion_statuses[shard] =
          wait_direct_batch(params, completion_statuses + shard);
      } else {
        continue;
      }
    }
    // The generic path is also a speculative-tail drain.  Completion errors
    // are handled as stale entries in the slot pass below; they must not
    // fail-stop the query or strand the exact critical retry.
    if (completion_statuses[shard] == 0 &&
        batch.issue_timestamp_ns[shard] != 0 &&
        speculative_completion_latency_ns != nullptr) {
      // The owner publishes the completion timestamp before the release
      // store to completion_status.  Consume that timestamp exactly once:
      // a nonblocking tail poll may observe one shard complete while another
      // shard in the same wave is still in flight.
      const u64 owner_completed_ns = completion_timestamps[shard];
      const u64 completed_ns =
        owner_completed_ns == 0 ? global_time_ns() : owner_completed_ns;
      atomicAdd(
        reinterpret_cast<unsigned long long*>(
          speculative_completion_latency_ns),
        static_cast<unsigned long long>(
          completed_ns - batch.issue_timestamp_ns[shard]));
      if (speculative_completion_groups != nullptr) {
        atomicAdd(
          reinterpret_cast<unsigned long long*>(
            speculative_completion_groups), 1ULL);
      }
      batch.issue_timestamp_ns[shard] = 0;
    }
  }
  __syncthreads();
  if (!wait_for_completion && speculative_finish_pending != 0) {
    return batch.fatal == 0;
  }

  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) *
      kPersistentFrontierRobCapacity + slot_begin;
  const u32* request_shards =
    params.speculative_graph_request_shards + request_base;
  for (u32 local = threadIdx.x; local < slot_count;
       local += blockDim.x) {
    const u32 slot = slot_begin + local;
    FrontierRobEntry& entry = rob[slot];
    if (entry.state !=
        static_cast<u8>(FrontierRequestState::inflight)) {
      continue;
    }
    const u32 shard = request_shards[local];
    if (shard == UINT32_MAX || completion_statuses[shard] != 0) {
      entry.state = static_cast<u8>(FrontierRequestState::stale);
      entry.validation =
        static_cast<u8>(FrontierValidationState::transport_rejected);
      if (speculative_stale != nullptr) {
        atomicAdd(speculative_stale, 1u);
      }
      if (classified_stale != nullptr) {
        atomicAdd(classified_stale, 1u);
      }
      if (speculative_wasted_bytes != nullptr &&
          entry.transfer_bytes != 0) {
        atomicAdd(
          reinterpret_cast<unsigned long long*>(
            speculative_wasted_bytes),
          static_cast<unsigned long long>(entry.transfer_bytes));
      }
      if (classified_wasted_bytes != nullptr &&
          entry.transfer_bytes != 0) {
        atomicAdd(
          reinterpret_cast<unsigned long long*>(classified_wasted_bytes),
          static_cast<unsigned long long>(entry.transfer_bytes));
      }
      continue;
    }
    entry.state = static_cast<u8>(FrontierRequestState::arrived);
    if (speculative_arrived != nullptr) {
      atomicAdd(speculative_arrived, 1u);
    }
    if (classified_arrived != nullptr) {
      atomicAdd(classified_arrived, 1u);
    }
    const FrontierValidationState validation =
      validate_frontier_record_local(
        params, descriptor.query_slot, entry, dynamic_telemetry);
    if (validation == FrontierValidationState::valid) {
      entry.state = static_cast<u8>(FrontierRequestState::validated);
      entry.validation =
        static_cast<u8>(FrontierValidationState::valid);
    } else {
      entry.state = static_cast<u8>(FrontierRequestState::stale);
      entry.validation = static_cast<u8>(validation);
      if (speculative_stale != nullptr) {
        atomicAdd(speculative_stale, 1u);
      }
      if (classified_stale != nullptr) {
        atomicAdd(classified_stale, 1u);
      }
      if (speculative_wasted_bytes != nullptr &&
          entry.transfer_bytes != 0) {
        atomicAdd(
          reinterpret_cast<unsigned long long*>(
            speculative_wasted_bytes),
          static_cast<unsigned long long>(entry.transfer_bytes));
      }
      if (classified_wasted_bytes != nullptr &&
          entry.transfer_bytes != 0) {
        atomicAdd(
          reinterpret_cast<unsigned long long*>(classified_wasted_bytes),
          static_cast<unsigned long long>(entry.transfer_bytes));
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && wait_for_completion &&
      classified_ready_waves != nullptr &&
      speculative_finish_pending == 0) {
    ++*classified_ready_waves;
  }
  if (threadIdx.x == 0) {
    *speculative_wait_cycles +=
      clock64() - speculative_wait_started_cycles;
    batch.active = 0;
    batch.fatal |= speculative_finish_failed;
  }
  __syncthreads();
  return batch.fatal == 0;
}

// Returns zero on success. A nonzero value is a failure detail consumed only
// by the query's fail-stop diagnostic. Preparation failures encode a reason
// in bits [3:0], the parent index in [8:4], and the destination scratch slot
// in [14:9]. Reason 1 is an unresolved handle, 2 is a stop request, 3 is a
// missing scratch arena, 4 is an out-of-range destination, and 6 is an
// otherwise impossible preparation rejection. Snapshot validation is reason
// 5. Bit 16 plus errno magnitude denotes an owner/direct completion failure.
__device__ u32 fetch_graph_records_batch(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    const u64* handles,
    u32 count,
    u32* acquired_slots,
    const u32* destination_scratch_slots,
    u32* remote_reads,
    u32* remote_batches,
    u32* graph_read_retries,
    u64* graph_read_bytes,
    u32* graph_live_extent_reads,
    u32* graph_full_record_reads,
    u32* graph_extent_fallback_reads,
    u32* graph_extent_underhint_reads,
    u32* graph_extent_hint_promotions,
    u32 route_attempt,
    u32 search_round,
    bool trace_enabled,
    GraphFetchCycleBreakdown* cycle_breakdown,
    DynamicGraphTelemetry* dynamic_telemetry = nullptr,
    const u8* force_full_underhint = nullptr) {
  __shared__ i32 shard_status[kPersistentMaxShards];
  __shared__ u64 shard_issue_timestamp_ns[kPersistentMaxShards];
  __shared__ u64 stage_started_cycles;
  __shared__ u32 trace_event_start;
  __shared__ u32 trace_batch_count;
  __shared__ u32 failed;
  __shared__ u32 retry_pending;
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) * kPersistentMaxMergeCandidates;
  u32* request_shards = params.dynamic_code_request_shards + request_base;
  u64* request_offsets = params.dynamic_code_request_offsets + request_base;
  u64* request_local_iovas =
    params.dynamic_code_request_local_iovas + request_base;
  u32* request_bytes = params.graph_request_bytes == nullptr
      ? nullptr : params.graph_request_bytes +
          static_cast<size_t>(descriptor.query_slot) *
            kPersistentMaxPrefetch;
  const bool live_extent_enabled = request_bytes != nullptr;
  const bool header_neighbor =
    params.graph_read_policy ==
      static_cast<u32>(GraphReadPolicy::header_neighbor);

  if (threadIdx.x == 0) failed = 0;
  for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
    acquired_slots[index] = UINT32_MAX;
    request_shards[index] = UINT32_MAX;
    request_offsets[index] = 0;
    request_local_iovas[index] = 0;
    if (request_bytes != nullptr) {
      request_bytes[index] = params.graph_entry_bytes;
    }
    remote_reads[index] = 0;
  }
  for (u32 shard = threadIdx.x; shard < params.num_shards; shard += blockDim.x) {
    shard_status[shard] = 0;
    shard_issue_timestamp_ns[shard] = 0;
  }
  __syncthreads();

  constexpr u32 warp_width = 32;
  // `remote_reads` is query-local scratch. Bit zero preserves its original
  // logical-read counter contract; the remaining bits carry retry provenance
  // across physical attempts without publishing an unverified short header.
  const u32 warp = threadIdx.x / warp_width;
  const u32 lane_in_warp = threadIdx.x % warp_width;
  const u32 warp_count = max(1u, blockDim.x / warp_width);
  if (lane_in_warp == 0) {
    for (u32 index = warp; index < count; index += warp_count) {
      u32 transfer_bytes = params.graph_entry_bytes;
      const u32 scratch_slot = destination_scratch_slots == nullptr
        ? index : destination_scratch_slots[index];
      if (!prepare_graph_record_in_scratch(
            params, handles[index], descriptor.query_slot,
            scratch_slot, acquired_slots[index],
            request_shards[index], request_offsets[index],
            request_local_iovas[index], transfer_bytes)) {
        // This branch is fail-stop only, so preserve the exact rejected
        // invariant without adding address resolution to successful queries.
        u32 reason =
          handles[index] == kInvalidDeviceHandle ? 7u
          : handles[index] == 0 ? 8u
          : 1u;
        if (scratch_slot >= kPersistentGraphScratchSlots) {
          reason = 4u;
        } else if (params.graph_scratch == nullptr) {
          reason = 3u;
        } else if (*reinterpret_cast<volatile u32*>(params.stop) != 0) {
          reason = 2u;
        } else {
          u64 raw = 0;
          u64 graph_offset = 0;
          u32 shard = 0;
          if (resolve_handle(
                params, handles[index], raw, shard, graph_offset)) {
            reason = 6u;
          }
        }
        const u32 detail =
          reason |
          ((index & 0x1fu) << 4) |
          ((min(scratch_slot, 0x3fu) & 0x3fu) << 9);
        atomicCAS(&failed, 0u, detail);
      } else {
        const bool continue_header_neighbor =
          header_neighbor && force_full_underhint != nullptr &&
          force_full_underhint[index] != 0;
        bool continue_header_neighbor_body = false;
        if (continue_header_neighbor) {
          const u8* prior_header = graph_record_pointer(
            params, descriptor.query_slot, acquired_slots[index]);
          u32 required_bytes = 0;
          if (graph_record_validation::required_live_extent_bytes(
                prior_header,
                graph_record_validation::kGraphRecordHeaderBytes,
                params.graph_degree, params.graph_entry_capacity,
                required_bytes) &&
              required_bytes >
                graph_record_validation::kGraphRecordHeaderBytes) {
            request_offsets[index] +=
              graph_record_validation::kGraphRecordHeaderBytes;
            request_local_iovas[index] +=
              graph_record_validation::kGraphRecordHeaderBytes;
            transfer_bytes = required_bytes -
              graph_record_validation::kGraphRecordHeaderBytes;
            continue_header_neighbor_body = true;
          }
        }
        const bool force_full = !header_neighbor &&
          force_full_underhint != nullptr &&
          force_full_underhint[index] != 0;
        remote_reads[index] = prepare_graph_read_attempt_state(
          params.graph_entry_bytes, force_full, transfer_bytes);
        if (continue_header_neighbor) {
          // The speculative header and this exact-prefix transfer form one
          // logical adjacency access.  Keep physical WQE/byte accounting for
          // both stages but do not count a second logical graph read.
          remote_reads[index] &= ~kGraphReadLogical;
          if (continue_header_neighbor_body) {
            remote_reads[index] |= kGraphReadHeaderNeighborBody;
          }
        }
        if (request_bytes != nullptr) {
          request_bytes[index] = transfer_bytes;
        }
      }
    }
  }
  __syncthreads();
  if (failed != 0) {
    for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
      acquired_slots[index] = UINT32_MAX;
    }
    __syncthreads();
    return failed;
  }

  // A compact graph entry is updated in-place by stage2/reverse-edge workers.
  // Its checksum is therefore an optimistic snapshot validator: a successful
  // RDMA completion can still overlap a legal publication and contain a torn
  // mix of the old and new entry. Storage CPU readers already retry this same
  // condition. Re-read only invalid entries and reserve fail-stop for a record
  // that remains invalid after the bounded snapshot attempts.
  constexpr u32 kFullGraphSnapshotAttempts = 3;
  // A failed optimistic short read is an extent-hint upgrade, not one of the
  // full-record snapshot retries available to the fixed path. Mixed batches
  // therefore need at most one short attempt plus the legacy three full
  // attempts. Entries that began full are still capped at exactly three.
  const u32 maximum_batch_attempts =
    live_extent_enabled ? kFullGraphSnapshotAttempts + 1u
                        : kFullGraphSnapshotAttempts;
  for (u32 attempt = 0; attempt < maximum_batch_attempts; ++attempt) {
    if (threadIdx.x == 0) retry_pending = 0;
    for (u32 shard = threadIdx.x; shard < params.num_shards;
         shard += blockDim.x) {
      shard_status[shard] = 0;
      shard_issue_timestamp_ns[shard] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      stage_started_cycles = clock64();
      trace_event_start = 0;
      trace_batch_count = 0;
      if (trace_enabled && params.query_rdma_trace_headers != nullptr) {
        for (u32 shard = 0; shard < params.num_shards; ++shard) {
          for (u32 index = 0; index < count; ++index) {
            if (request_shards[index] == shard) {
              ++trace_batch_count;
              break;
            }
          }
        }
        QueryRdmaTraceHeader& header =
          params.query_rdma_trace_headers[descriptor.query_slot];
        trace_event_start = header.event_count;
        header.event_count += trace_batch_count;
        if (header.event_count > params.query_rdma_trace_events_per_query) {
          header.overflow = 1;
        }
      }
    }
    __syncthreads();
    for (u32 shard = threadIdx.x; shard < params.num_shards;
         shard += blockDim.x) {
      u32 matching = 0;
      u32 payload_bytes = 0;
      u32 minimum_bytes = UINT32_MAX;
      u32 maximum_bytes = 0;
      u32 short_reads = 0;
      u32 full_reads = 0;
      u32 fallback_reads = 0;
      u32 underhint_reads = 0;
      u32 retry_reads = 0;
      u32 dynamic_short_reads = 0;
      u32 dynamic_full_reads = 0;
      u32 dynamic_fallback_reads = 0;
      u64 dynamic_payload_bytes = 0;
      for (u32 index = 0; index < count; ++index) {
        if (request_shards[index] != shard) continue;
        ++matching;
        const u32 transfer_bytes = live_extent_enabled
          ? request_bytes[index] : params.graph_entry_bytes;
        const bool full_record_read =
          transfer_bytes >= params.graph_entry_bytes;
        const GraphReadAdmissionAccounting admission =
          classify_graph_read_admission(
            remote_reads[index], full_record_read, attempt);
        retry_reads += admission.retry_reads;
        if (live_extent_enabled) {
          payload_bytes += transfer_bytes;
          minimum_bytes = min(minimum_bytes, transfer_bytes);
          maximum_bytes = max(maximum_bytes, transfer_bytes);
          if (transfer_bytes < params.graph_entry_bytes) {
            ++short_reads;
          } else {
            ++full_reads;
            fallback_reads += admission.fallback_reads;
            underhint_reads += admission.underhint_reads;
          }
        }
        if (dynamic_graph_telemetry_handle(handles[index])) {
          dynamic_payload_bytes += transfer_bytes;
          if (transfer_bytes < params.graph_entry_bytes) {
            ++dynamic_short_reads;
          } else {
            ++dynamic_full_reads;
            dynamic_fallback_reads += admission.fallback_reads;
          }
        }
      }
      if (!live_extent_enabled && matching != 0) {
        payload_bytes = matching * params.graph_entry_bytes;
        minimum_bytes = params.graph_entry_bytes;
        maximum_bytes = params.graph_entry_bytes;
        full_reads = matching;
      }
      if (matching != 0) {
        if (trace_enabled && params.query_rdma_trace_events != nullptr) {
          u32 ordinal = 0;
          for (u32 prior = 0; prior < shard; ++prior) {
            for (u32 index = 0; index < count; ++index) {
              if (request_shards[index] == prior) {
                ++ordinal;
                break;
              }
            }
          }
          const u32 event_index = trace_event_start + ordinal;
          if (event_index < params.query_rdma_trace_events_per_query) {
            params.query_rdma_trace_events[
              static_cast<size_t>(descriptor.query_slot) *
                params.query_rdma_trace_events_per_query + event_index] = {
                  .request_id = descriptor.request_id,
                  .issue_timestamp_ns = global_time_ns(),
                  .route_attempt = route_attempt,
                  .search_round = search_round,
                  .snapshot_attempt = attempt,
                  .target_shard = shard,
                  .parent_count = matching,
                  .payload_bytes = payload_bytes,
                  .minimum_bytes_per_parent = minimum_bytes,
                  .maximum_bytes_per_parent = maximum_bytes,
                };
          }
        }
      }
      i32* owner_completion = params.direct_batch_statuses == nullptr ? nullptr :
        params.direct_batch_statuses +
          static_cast<size_t>(descriptor.query_slot) * params.num_shards + shard;
      u64* owner_completion_timestamp_ns =
        !trace_enabled ||
        params.direct_batch_completion_timestamps_ns == nullptr ? nullptr :
          params.direct_batch_completion_timestamps_ns +
            static_cast<size_t>(descriptor.query_slot) * params.num_shards +
            shard;
      shard_issue_timestamp_ns[shard] = global_time_ns();
      shard_status[shard] = direct_fetch_batch(
          params, shard, request_shards, request_offsets, count,
          params.graph_scratch, kPersistentGraphReadBytes,
          params.graph_entry_bytes,
          (descriptor.query_slot + shard) % params.direct_qps_per_node,
          request_local_iovas, owner_completion, true, nullptr,
          owner_completion_timestamp_ns,
          live_extent_enabled ? request_bytes : nullptr);
      // Account only work that completed synchronously or was successfully
      // admitted to an owner queue. Parameter/enqueue/transport failures that
      // return before admission did not issue the advertised graph payload.
      const bool admitted =
        shard_status[shard] == 0 ||
        shard_status[shard] == -EINPROGRESS;
      if (matching != 0 && admitted) {
        atomicAdd(remote_batches, 1u);
        if (graph_read_bytes != nullptr) {
          atomicAdd(
            reinterpret_cast<unsigned long long*>(graph_read_bytes),
            static_cast<unsigned long long>(payload_bytes));
        }
        if (graph_live_extent_reads != nullptr && short_reads != 0) {
          atomicAdd(graph_live_extent_reads, short_reads);
        }
        if (graph_full_record_reads != nullptr && full_reads != 0) {
          atomicAdd(graph_full_record_reads, full_reads);
        }
        if (graph_extent_fallback_reads != nullptr &&
            fallback_reads != 0) {
          atomicAdd(graph_extent_fallback_reads, fallback_reads);
        }
        if (graph_extent_underhint_reads != nullptr &&
            underhint_reads != 0) {
          atomicAdd(graph_extent_underhint_reads, underhint_reads);
        }
        add_dynamic_graph_read_telemetry(
          dynamic_telemetry, dynamic_short_reads, dynamic_full_reads,
          dynamic_payload_bytes, dynamic_fallback_reads);
        // Count one fallback only when its first full WQE is admitted. Further
        // full snapshot retries remain graph_read_retries but are not another
        // stale-hint fallback.
        if (fallback_reads != 0) {
          for (u32 index = 0; index < count; ++index) {
            if (request_shards[index] == shard) {
              remote_reads[index] =
                graph_read_state_after_fallback_admission(
                  remote_reads[index]);
            }
          }
        }
        if (graph_read_retries != nullptr) {
          // A query-local force-full is attempt zero only inside this helper;
          // physically it follows the already-admitted async short and must
          // retain the original short->full retry count. Later checksum full
          // attempts continue to contribute once each.
          if (retry_reads != 0) {
            atomicAdd(graph_read_retries, retry_reads);
          }
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      if (trace_enabled && params.query_rdma_trace_events != nullptr) {
        const u64 wait_phase_start_ns = global_time_ns();
        const u32 available = trace_event_start >=
            params.query_rdma_trace_events_per_query ? 0 :
          min(trace_batch_count,
              params.query_rdma_trace_events_per_query - trace_event_start);
        for (u32 ordinal = 0; ordinal < available; ++ordinal) {
          QueryRdmaTraceEvent& event = params.query_rdma_trace_events[
            static_cast<size_t>(descriptor.query_slot) *
              params.query_rdma_trace_events_per_query +
            trace_event_start + ordinal];
          event.wait_phase_start_timestamp_ns = wait_phase_start_ns;
        }
      }
      if (cycle_breakdown != nullptr) {
        cycle_breakdown->issue += clock64() - stage_started_cycles;
        stage_started_cycles = clock64();
      }
    }
    __syncthreads();
    for (u32 shard = threadIdx.x; shard < params.num_shards;
         shard += blockDim.x) {
      if (shard_status[shard] != -EINPROGRESS) continue;
      i32* owner_completion = params.direct_batch_statuses +
        static_cast<size_t>(descriptor.query_slot) * params.num_shards + shard;
      shard_status[shard] = wait_direct_batch(params, owner_completion);
    }
    __syncthreads();
    for (u32 shard = threadIdx.x; shard < params.num_shards;
         shard += blockDim.x) {
      if (shard_status[shard] != 0 ||
          shard_issue_timestamp_ns[shard] == 0 ||
          cycle_breakdown == nullptr) {
        continue;
      }
      const u64 completion_timestamp_ns = global_time_ns();
      atomicAdd(
        reinterpret_cast<unsigned long long*>(
          &cycle_breakdown->completion_latency_ns),
        static_cast<unsigned long long>(
          completion_timestamp_ns - shard_issue_timestamp_ns[shard]));
      atomicAdd(
        reinterpret_cast<unsigned long long*>(
          &cycle_breakdown->completion_groups), 1ULL);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      if (cycle_breakdown != nullptr) {
        cycle_breakdown->wait += clock64() - stage_started_cycles;
      }
      const u64 process_start_ns = trace_enabled ? global_time_ns() : 0;
      if (trace_enabled && params.query_rdma_trace_headers != nullptr &&
          params.query_rdma_trace_events != nullptr) {
        const u32 available = trace_event_start >=
            params.query_rdma_trace_events_per_query ? 0 :
          min(trace_batch_count,
              params.query_rdma_trace_events_per_query - trace_event_start);
        for (u32 ordinal = 0; ordinal < available; ++ordinal) {
          QueryRdmaTraceEvent& event = params.query_rdma_trace_events[
            static_cast<size_t>(descriptor.query_slot) *
              params.query_rdma_trace_events_per_query +
            trace_event_start + ordinal];
          event.completion_timestamp_ns =
            params.direct_batch_completion_timestamps_ns[
              static_cast<size_t>(descriptor.query_slot) *
                params.num_shards + event.target_shard];
          event.batch_process_start_timestamp_ns = process_start_ns;
        }
      }
      stage_started_cycles = clock64();
    }
    __syncthreads();

    for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
      const u32 shard = request_shards[index];
      if (shard == UINT32_MAX) continue;
      const u32 slot = acquired_slots[index];
      u8* record = graph_record_pointer(params, descriptor.query_slot, slot);
      const i32 status = shard_status[shard];
      const u32 physical_transfer_bytes = live_extent_enabled
        ? request_bytes[index] : params.graph_entry_bytes;
      const bool header_neighbor_body = header_neighbor &&
        (remote_reads[index] & kGraphReadHeaderNeighborBody) != 0;
      const u32 available_bytes = physical_transfer_bytes +
        (header_neighbor_body
          ? graph_record_validation::kGraphRecordHeaderBytes : 0u);
      const bool partial_read = available_bytes < params.graph_entry_bytes;
      u32 required_bytes = 0;
      const bool prefix_valid = status == 0 &&
        graph_record_validation::required_live_extent_bytes(
          record, available_bytes, params.graph_degree,
          params.graph_entry_capacity, required_bytes);
      // Capacity is checked before checksum acceptance. This prevents a
      // truncated counted prefix from being accepted even in the event of a
      // checksum collision. Any other invalid reconstructed short read also
      // upgrades to the authoritative full record rather than repeating the
      // same insufficient request.
      const bool short_read_requires_full =
        status == 0 && partial_read &&
        (!prefix_valid || required_bytes > available_bytes);
      const bool extent_underhint =
        status == 0 && partial_read && prefix_valid &&
        required_bytes > available_bytes;
      const graph_record_validation::SnapshotState snapshot =
        status == 0 && !short_read_requires_full
          ? (partial_read
              ? classify_short_graph_record(
                  params, record, available_bytes, handles[index])
              : classify_graph_record(params, record, handles[index]))
          : graph_record_validation::SnapshotState::invalid;
      const bool started_with_short =
        (remote_reads[index] & kGraphReadStartedWithShortExtent) != 0;
      const bool attempts_remain =
        graph_record_validation::snapshot_retry_available(
          attempt, started_with_short, partial_read,
          maximum_batch_attempts, kFullGraphSnapshotAttempts);
      const graph_record_validation::ReadAction action =
        short_read_requires_full
          ? (attempts_remain
              ? graph_record_validation::ReadAction::retry
              : graph_record_validation::ReadAction::fail)
          : graph_record_validation::decide_read_action(
              status == 0, snapshot, attempts_remain);
      if (action == graph_record_validation::ReadAction::accept) {
        // Learn only from the checksum-valid authoritative full snapshot, not
        // from the optimistic short header that triggered the upgrade. This
        // prevents a transient torn header from inflating the high-water hint.
        if (!partial_read &&
            (remote_reads[index] & kGraphReadExtentUnderhint) != 0 &&
            promote_graph_extent_class(
              params, handles[index], required_bytes)) {
          if (graph_extent_hint_promotions != nullptr) {
            atomicAdd(graph_extent_hint_promotions, 1u);
          }
          if (dynamic_telemetry != nullptr &&
              dynamic_graph_telemetry_handle(handles[index])) {
            atomicAdd(&dynamic_telemetry->hint_promotions, 1u);
          }
        }
        if (adapt_dynamic_graph_extent_class(
              params, handles[index], required_bytes) ==
              DynamicGraphExtentAdaptation::demoted &&
            dynamic_telemetry != nullptr) {
          atomicAdd(&dynamic_telemetry->hint_demotions, 1u);
        }
        // UINT32_MAX removes this entry from subsequent per-shard retry
        // batches while leaving acquired_slots intact for traversal.
        request_shards[index] = UINT32_MAX;
        continue;
      }
      if (action == graph_record_validation::ReadAction::discard_stale) {
        // Slot reuse is an ordinary read-committed race.  The tagged handle no
        // longer names this record, so drop only this expansion; disabling the
        // direct path would turn normal update churn into a service failure.
        acquired_slots[index] = UINT32_MAX;
        request_shards[index] = UINT32_MAX;
        continue;
      }
      if (action == graph_record_validation::ReadAction::retry) {
        if (header_neighbor) {
          if (header_neighbor_body) {
            request_offsets[index] -=
              graph_record_validation::kGraphRecordHeaderBytes;
            request_local_iovas[index] -=
              graph_record_validation::kGraphRecordHeaderBytes;
            remote_reads[index] &= ~kGraphReadHeaderNeighborBody;
            request_bytes[index] =
              graph_record_validation::kGraphRecordHeaderBytes;
          } else if (extent_underhint) {
            request_offsets[index] +=
              graph_record_validation::kGraphRecordHeaderBytes;
            request_local_iovas[index] +=
              graph_record_validation::kGraphRecordHeaderBytes;
            request_bytes[index] = required_bytes -
              graph_record_validation::kGraphRecordHeaderBytes;
            remote_reads[index] |= kGraphReadHeaderNeighborBody;
          } else {
            request_bytes[index] =
              graph_record_validation::kGraphRecordHeaderBytes;
          }
        } else if (partial_read) {
            request_bytes[index] = params.graph_entry_bytes;
            remote_reads[index] |= kGraphReadNeedsExtentFallback;
            if (extent_underhint) {
              remote_reads[index] |= kGraphReadExtentUnderhint;
            }
        }
        atomicAdd(&retry_pending, 1u);
        continue;
      }

      acquired_slots[index] = UINT32_MAX;
      request_shards[index] = UINT32_MAX;
      const u32 status_magnitude = status < 0
        ? min(static_cast<u32>(-(status + 1)) + 1u, 0xffffu)
        : min(static_cast<u32>(status), 0xffffu);
      atomicCAS(
        &failed, 0u,
        status == 0 ? 5u : (u32{1} << 16) | status_magnitude);
      if (status == 0) {
        if (params.direct_error != nullptr) {
          atomicCAS(params.direct_error, 0, -EBADMSG);
        }
        atomicExch(params.direct_disabled, 1u);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0 && cycle_breakdown != nullptr) {
      cycle_breakdown->validation += clock64() - stage_started_cycles;
    }
    if (retry_pending == 0) break;
    if (threadIdx.x == 0) device_ring_relax(128);
    __syncthreads();
  }
  return failed;
}

}  // namespace gpu_search::persistent_kernel_detail
