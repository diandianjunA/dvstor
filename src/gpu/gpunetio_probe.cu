#include "gpu/gpunetio_probe.hh"

#include <cuda_runtime.h>

#include <cerrno>
#include <limits>
#include <stdexcept>

#ifndef IBV_WC_DRIVER1
#define IBV_WC_DRIVER1 135
#define IBV_WC_DRIVER2 136
#define IBV_WC_DRIVER3 137
#endif

#include <doca_gpunetio_dev_verbs_onesided.cuh>

namespace gpu {

namespace {

constexpr uint64_t kPollSpinLimit = 100000000ULL;
constexpr int kPollTimeoutStatus = -110;

__device__ __forceinline__ uint64_t probe_global_time_ns() {
  uint64_t value = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ uint64_t probe_mix64(uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode sharing_mode>
__device__ inline int poll_cq_at_with_timeout(struct doca_gpu_dev_verbs_cq* cq,
                                              const uint64_t ticket,
                                              uint64_t* cqe_debug) {
  auto* cqe_base = reinterpret_cast<struct mlx5_cqe64*>(__ldg((uintptr_t*)&cq->cqe_daddr));
  const uint32_t cqe_num = __ldg(&cq->cqe_num);
  const uint32_t idx = ticket & (cqe_num - 1);
  auto* cqe64 = &cqe_base[idx];

  uint64_t curr_cons_index = 0;
  uint8_t opown = 0;
  for (uint64_t spins = 0; spins < kPollSpinLimit; ++spins) {
    curr_cons_index =
      doca_gpu_dev_verbs_load_relaxed<sharing_mode>(&cq->cqe_ci);
    opown = doca_gpu_dev_verbs_load_relaxed_sys_global(reinterpret_cast<uint8_t*>(&cqe64->op_own));
    if (!((curr_cons_index <= ticket) && ((opown & MLX5_CQE_OWNER_MASK) ^ !!(ticket & cqe_num)))) {
      const uint8_t opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;
      doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
      doca_gpu_dev_verbs_cq_update_dbrec<false>(cq, 1);
      return opcode == MLX5_CQE_REQ ? 0 : -EIO;
    }
  }

  if (cqe_debug != nullptr) {
    cqe_debug[0] = 0x54494d454f5554ULL;
    cqe_debug[1] = (ticket << 32) | idx;
    cqe_debug[2] = curr_cons_index;
    cqe_debug[3] = opown;
  }

  return kPollTimeoutStatus;
}

__device__ inline int poll_payload_cq_at_with_timeout(
    struct doca_gpu_dev_verbs_cq* cq, const uint64_t ticket,
    const uint64_t timeout_ns) {
  auto* cqe_base = reinterpret_cast<struct mlx5_cqe64*>(
    __ldg(reinterpret_cast<uintptr_t*>(&cq->cqe_daddr)));
  const uint32_t cqe_num = __ldg(&cq->cqe_num);
  const uint32_t idx = ticket & (cqe_num - 1);
  auto* cqe64 = &cqe_base[idx];
  const uint64_t started = probe_global_time_ns();

  for (;;) {
    const uint64_t curr_cons_index =
      doca_gpu_dev_verbs_load_relaxed<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(&cq->cqe_ci);
    const uint8_t opown = doca_gpu_dev_verbs_load_relaxed_sys_global(
      reinterpret_cast<uint8_t*>(&cqe64->op_own));
    if (!((curr_cons_index <= ticket) &&
          ((opown & MLX5_CQE_OWNER_MASK) ^ !!(ticket & cqe_num)))) {
      const uint8_t opcode =
        opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;
      doca_gpu_dev_verbs_fence_acquire<
        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
      doca_gpu_dev_verbs_cq_update_dbrec<false>(cq, 1);
      return opcode == MLX5_CQE_REQ ? 0 : -EIO;
    }
    if (probe_global_time_ns() - started >= timeout_ns) {
      return kPollTimeoutStatus;
    }
    __nanosleep(128);
  }
}

__global__ void gpunetio_read_probe_kernel(GpuNetioReadProbeParams params) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (params.remote_region_count == 0 || params.remote_regions == nullptr ||
      params.qp_array == nullptr ||
      params.qp_index >= params.qp_count ||
      params.remote_region >= params.remote_region_count ||
      params.qp_array[params.qp_index] == nullptr) {
    *params.status_code = -EINVAL;
    return;
  }
  const GpuNetioRemoteMemoryRegion& region = params.remote_regions[params.remote_region];
  if (region.address == 0 || region.rkey == 0 ||
      region.bytes < sizeof(uint64_t) ||
      region.address > UINT64_MAX - (region.bytes - 1u)) {
    *params.status_code = -ERANGE;
    return;
  }
  params.debug_values[0] = region.address;
  params.debug_values[1] = region.rkey;
  params.debug_values[2] = reinterpret_cast<uint64_t>(params.destination) - params.local_iova_base;
  params.debug_values[3] = params.local_mkey;
  auto* qp = reinterpret_cast<doca_gpu_dev_verbs_qp*>(params.qp_array[params.qp_index]);
  auto* cq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
  params.debug_values[5] = static_cast<uint64_t>(qp->need_dump) |
                           (static_cast<uint64_t>(qp->nic_handler) << 8) |
                           (static_cast<uint64_t>(qp->mem_type) << 16);
  params.debug_values[10] = qp->sq_rsvd_index;
  params.debug_values[11] = qp->sq_ready_index;
  params.debug_values[12] = qp->sq_wqe_pi;
  params.debug_values[13] = reinterpret_cast<uint64_t>(qp->sq_wqe_daddr);
  params.debug_values[14] = reinterpret_cast<uint64_t>(qp->sq_dbrec);
  params.debug_values[15] = reinterpret_cast<uint64_t>(qp->sq_db);
  params.debug_values[16] = reinterpret_cast<uint64_t>(cq->cqe_daddr);
  params.debug_values[17] = static_cast<uint64_t>(qp->sq_wqe_num) |
                            (static_cast<uint64_t>(qp->sq_wqe_mask) << 16) |
                            (static_cast<uint64_t>(cq->cqe_num) << 32);
  doca_gpu_dev_verbs_ticket_t ticket =
    doca_gpu_dev_verbs_atomic_read<uint64_t,
      DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&qp->sq_wqe_pi);
  auto* probe_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, ticket);
  doca_gpu_dev_verbs_wqe_prepare_read(
    qp,
    probe_wqe_ptr,
    ticket,
    DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
    region.address,
    region.rkey,
    reinterpret_cast<uint64_t>(params.destination) - params.local_iova_base,
    params.local_mkey,
    sizeof(uint64_t));
  doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
    qp, ticket + 1);
  params.debug_values[4] = ticket;
  params.debug_values[18] = qp->sq_rsvd_index;
  params.debug_values[19] = qp->sq_ready_index;
  params.debug_values[20] = qp->sq_wqe_pi;
  params.debug_values[21] = *qp->sq_dbrec;
  const auto* probe_wqe = reinterpret_cast<const uint64_t*>(
    qp->sq_wqe_daddr + ((ticket & qp->sq_wqe_mask) << DOCA_GPUNETIO_MLX5_WQE_SQ_SHIFT));
  for (uint32_t i = 0; i < 8; ++i) {
    params.debug_values[22 + i] = probe_wqe[i];
    params.debug_values[30 + i] = probe_wqe[i];
  }
  const int status = poll_cq_at_with_timeout<
    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
      cq, ticket, params.debug_values + 6);
  params.debug_values[39] = static_cast<uint32_t>(status);
  if (status != 0) {
    params.debug_values[38] = 1;
    *params.status_code = status;
    return;
  }

  const doca_gpu_dev_verbs_ticket_t read_ticket = qp->sq_wqe_pi;
  auto* read_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, read_ticket);
  doca_gpu_dev_verbs_wqe_prepare_read(
    qp,
    read_wqe_ptr,
    read_ticket,
    DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
    region.address,
    region.rkey,
    reinterpret_cast<uint64_t>(params.destination) - params.local_iova_base,
    params.local_mkey,
    sizeof(uint64_t));
  doca_gpu_dev_verbs_ticket_t final_ticket = read_ticket;
  if (qp->need_dump) {
    final_ticket = read_ticket + 1;
    auto* dump_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, final_ticket);
    doca_gpu_dev_verbs_wqe_prepare_dump(
      qp,
      dump_wqe_ptr,
      final_ticket,
      DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
      reinterpret_cast<uint64_t>(params.dump_ptr) - params.local_iova_base,
      params.local_mkey,
      1);
  }
  doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
    qp, final_ticket + 1);
  int dump_status = poll_cq_at_with_timeout<
    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
      cq, read_ticket, params.debug_values + 6);
  if (dump_status == 0 && final_ticket != read_ticket) {
    dump_status = poll_cq_at_with_timeout<
      DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
        cq, final_ticket, params.debug_values + 6);
  }
  params.debug_values[4] = final_ticket;
  params.debug_values[18] = qp->sq_rsvd_index;
  params.debug_values[19] = qp->sq_ready_index;
  params.debug_values[20] = qp->sq_wqe_pi;
  params.debug_values[21] = *qp->sq_dbrec;
  params.debug_values[38] = 2;
  params.debug_values[39] = static_cast<uint32_t>(dump_status);
  *params.status_code = dump_status;
}

__global__ void gpunetio_payload_probe_kernel(
    GpuNetioPayloadProbeParams params) {
  constexpr uint32_t kWarpWidth = 32;
  const uint32_t lane = threadIdx.x % kWarpWidth;
  const uint32_t warps_per_block = blockDim.x / kWarpWidth;
  const uint32_t warp_in_block = threadIdx.x / kWarpWidth;
  const uint32_t worker = blockIdx.x * warps_per_block + warp_in_block;
  if (worker >= params.active_qps) return;
  const uint64_t combined_stage_bytes =
    static_cast<uint64_t>(params.first_stage_bytes) +
    params.second_stage_bytes;

  int status = 0;
  if (lane == 0) {
    if (params.remote_region_count == 0 || params.qp_array == nullptr ||
        params.remote_regions == nullptr || worker >= params.qp_count ||
        params.qp_array[worker] == nullptr || params.destination == nullptr ||
        params.dump_ptr == nullptr || params.first_stage_bytes == 0 ||
        params.batch_reads == 0 || params.measured_batches == 0 ||
        params.warmup_batches > UINT32_MAX - params.measured_batches ||
        params.destination_stride < params.first_stage_bytes ||
        (params.second_stage_bytes != 0 &&
         static_cast<uint64_t>(params.destination_stride) <
           combined_stage_bytes) ||
        static_cast<uint64_t>(params.remote_record_stride) <
          combined_stage_bytes ||
        params.remote_span_bytes < params.remote_record_stride ||
        params.first_stage_bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE ||
        params.second_stage_bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      status = -EINVAL;
    }
    params.status_codes[worker] = status;
    params.completed_reads[worker] = 0;
    params.dump_wqe_flags[worker] = 0;
  }
  status = __shfl_sync(0xffffffffu, status, 0);
  if (status != 0) return;

  const uint32_t memory_node = worker % params.remote_region_count;
  const GpuNetioRemoteMemoryRegion region =
    params.remote_regions[memory_node];
  if (lane == 0 &&
      (region.address == 0 || region.rkey == 0 || region.bytes == 0 ||
       region.address > UINT64_MAX - (region.bytes - 1u) ||
       region.bytes < params.remote_span_bytes)) {
    status = -ERANGE;
    params.status_codes[worker] = status;
  }
  status = __shfl_sync(0xffffffffu, status, 0);
  if (status != 0) return;
  auto* qp =
    reinterpret_cast<doca_gpu_dev_verbs_qp*>(params.qp_array[worker]);
  auto* cq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
  const bool need_dump = qp->need_dump;
  const uint32_t stage_count = params.second_stage_bytes == 0 ? 1u : 2u;
  if (lane == 0) {
    params.dump_wqe_flags[worker] = need_dump ? 1u : 0u;
    if (static_cast<uint64_t>(params.batch_reads) +
          (need_dump ? 1u : 0u) > qp->sq_wqe_num) {
      status = -E2BIG;
      params.status_codes[worker] = status;
    }
  }
  status = __shfl_sync(0xffffffffu, status, 0);
  if (status != 0) return;

  uint64_t measured_reads = 0;
  const uint32_t total_batches =
    params.warmup_batches + params.measured_batches;
  for (uint32_t batch = 0; batch < total_batches; ++batch) {
    uint64_t batch_started_ns = 0;
    if (lane == 0) batch_started_ns = probe_global_time_ns();

    for (uint32_t stage = 0; stage < stage_count; ++stage) {
      const uint32_t bytes =
        stage == 0 ? params.first_stage_bytes : params.second_stage_bytes;
      const uint32_t remote_stage_offset =
        stage == 0 ? 0u : params.first_stage_bytes;
      uint64_t first_wqe = 0;
      uint64_t completion_ticket = 0;
      if (lane == 0) {
        first_wqe = qp->sq_wqe_pi;
        completion_ticket =
          doca_gpu_dev_verbs_load_relaxed<
            DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
              &cq->cqe_ci);
      }
      first_wqe = __shfl_sync(0xffffffffu, first_wqe, 0);
      completion_ticket =
        __shfl_sync(0xffffffffu, completion_ticket, 0);

      for (uint32_t read = lane; read < params.batch_reads;
           read += kWarpWidth) {
        const uint64_t ticket = first_wqe + read;
        auto* wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, ticket);
        const bool final_read = read + 1 == params.batch_reads;
        const auto flags = !need_dump && final_read
          ? DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE
          : DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;
        const uint64_t record_count =
          params.remote_span_bytes / params.remote_record_stride;
        const uint64_t logical_read =
          (static_cast<uint64_t>(worker) << 40) ^
          (static_cast<uint64_t>(batch) << 16) ^ read;
        const uint64_t record =
          probe_mix64(logical_read + 0x9e3779b97f4a7c15ULL) %
          record_count;
        const uint64_t remote_address =
          region.address + record * params.remote_record_stride +
          remote_stage_offset;
        auto* destination =
          params.destination +
          (static_cast<size_t>(worker) * params.batch_reads + read) *
            params.destination_stride + remote_stage_offset;
        const uint64_t local_iova =
          reinterpret_cast<uint64_t>(destination) - params.local_iova_base;
        doca_gpu_dev_verbs_wqe_prepare_read(
          qp, wqe, ticket, flags, remote_address, region.rkey,
          local_iova, params.local_mkey, bytes);
      }
      __syncwarp();

      if (lane == 0) {
        uint64_t final_wqe = first_wqe + params.batch_reads - 1;
        if (need_dump) {
          final_wqe = first_wqe + params.batch_reads;
          auto* dump_wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, final_wqe);
          doca_gpu_dev_verbs_wqe_prepare_dump(
            qp, dump_wqe, final_wqe,
            DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
            reinterpret_cast<uint64_t>(params.dump_ptr) -
              params.local_iova_base,
            params.local_mkey, 1);
        }
        doca_gpu_dev_verbs_submit<
          DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
            qp, final_wqe + 1);
        status = poll_payload_cq_at_with_timeout(
          cq, completion_ticket, params.timeout_ns);
      }
      status = __shfl_sync(0xffffffffu, status, 0);
      __syncwarp();
      if (status != 0) break;
    }

    if (lane == 0 && status == 0 && batch >= params.warmup_batches) {
      const uint32_t measured_index = batch - params.warmup_batches;
      params.batch_latency_ns[
        static_cast<size_t>(worker) * params.measured_batches +
        measured_index] = probe_global_time_ns() - batch_started_ns;
      measured_reads +=
        static_cast<uint64_t>(params.batch_reads) * stage_count;
    }
    status = __shfl_sync(0xffffffffu, status, 0);
    if (status != 0) break;
  }

  if (lane == 0) {
    params.status_codes[worker] = status;
    params.completed_reads[worker] = measured_reads;
  }
}

}  // namespace

void launch_gpunetio_read_probe(
    cudaStream_t stream, const GpuNetioReadProbeParams& params) {
  const uint64_t destination = reinterpret_cast<uint64_t>(params.destination);
  const uint64_t dump = reinterpret_cast<uint64_t>(params.dump_ptr);
  if (params.remote_regions == nullptr || params.remote_region_count == 0 ||
      params.qp_array == nullptr || params.qp_count == 0 ||
      params.qp_index >= params.qp_count ||
      params.remote_region >= params.remote_region_count ||
      params.destination == nullptr || params.dump_ptr == nullptr ||
      params.status_code == nullptr || params.debug_values == nullptr ||
      destination < params.local_iova_base || dump < params.local_iova_base ||
      reinterpret_cast<uintptr_t>(params.remote_regions) %
          alignof(GpuNetioRemoteMemoryRegion) != 0 ||
      reinterpret_cast<uintptr_t>(params.qp_array) % alignof(void*) != 0 ||
      reinterpret_cast<uintptr_t>(params.status_code) % alignof(int) != 0 ||
      reinterpret_cast<uintptr_t>(params.debug_values) % alignof(uint64_t) != 0) {
    throw std::invalid_argument("invalid GPUNetIO read-probe parameters");
  }
  gpunetio_read_probe_kernel<<<1, 1, 0, stream>>>(params);
}

void launch_gpunetio_payload_probe(
    cudaStream_t stream, const GpuNetioPayloadProbeParams& params) {
  constexpr uint32_t kThreads = 128;
  constexpr uint32_t kWarpsPerBlock = kThreads / 32;
  const uint64_t combined_stage_bytes =
    static_cast<uint64_t>(params.first_stage_bytes) +
    params.second_stage_bytes;
  const uint64_t destination = reinterpret_cast<uint64_t>(params.destination);
  const uint64_t dump = reinterpret_cast<uint64_t>(params.dump_ptr);
  const uint64_t destination_bytes_per_worker =
    static_cast<uint64_t>(params.batch_reads) * params.destination_stride;
  const uint64_t latency_entries =
    static_cast<uint64_t>(params.active_qps) * params.measured_batches;
  const uint32_t stage_count = params.second_stage_bytes == 0 ? 1u : 2u;
  const uint64_t reads_per_batch =
    static_cast<uint64_t>(params.batch_reads) * stage_count;
  if (params.remote_regions == nullptr || params.remote_region_count == 0 ||
      params.qp_array == nullptr || params.qp_count == 0 ||
      params.active_qps == 0 || params.active_qps > params.qp_count ||
      params.destination == nullptr || params.dump_ptr == nullptr ||
      params.status_codes == nullptr || params.completed_reads == nullptr ||
      params.dump_wqe_flags == nullptr || params.batch_latency_ns == nullptr ||
      params.first_stage_bytes == 0 || params.batch_reads == 0 ||
      params.measured_batches == 0 || params.timeout_ns == 0 ||
      params.warmup_batches >
        std::numeric_limits<uint32_t>::max() - params.measured_batches ||
      params.first_stage_bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE ||
      params.second_stage_bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE ||
      combined_stage_bytes > params.destination_stride ||
      combined_stage_bytes > params.remote_record_stride ||
      params.remote_span_bytes < params.remote_record_stride ||
      destination < params.local_iova_base || dump < params.local_iova_base ||
      destination_bytes_per_worker >
        std::numeric_limits<size_t>::max() / params.active_qps ||
      latency_entries >
        std::numeric_limits<size_t>::max() / sizeof(uint64_t) ||
      params.measured_batches >
        std::numeric_limits<uint64_t>::max() / reads_per_batch ||
      reinterpret_cast<uintptr_t>(params.remote_regions) %
          alignof(GpuNetioRemoteMemoryRegion) != 0 ||
      reinterpret_cast<uintptr_t>(params.qp_array) % alignof(void*) != 0 ||
      reinterpret_cast<uintptr_t>(params.status_codes) % alignof(int) != 0 ||
      reinterpret_cast<uintptr_t>(params.completed_reads) %
          alignof(uint64_t) != 0 ||
      reinterpret_cast<uintptr_t>(params.dump_wqe_flags) %
          alignof(uint32_t) != 0 ||
      reinterpret_cast<uintptr_t>(params.batch_latency_ns) %
          alignof(uint64_t) != 0) {
    throw std::invalid_argument("invalid GPUNetIO payload-probe parameters");
  }
  const uint64_t destination_span =
    destination_bytes_per_worker * params.active_qps;
  if (destination > std::numeric_limits<uintptr_t>::max() - destination_span ||
      dump == std::numeric_limits<uintptr_t>::max()) {
    throw std::overflow_error("GPUNetIO payload-probe pointer range overflows");
  }
  const uint64_t blocks_wide =
    (static_cast<uint64_t>(params.active_qps) + kWarpsPerBlock - 1u) /
    kWarpsPerBlock;
  const uint32_t blocks = static_cast<uint32_t>(blocks_wide);
  gpunetio_payload_probe_kernel<<<blocks, kThreads, 0, stream>>>(params);
}

}  // namespace gpu
