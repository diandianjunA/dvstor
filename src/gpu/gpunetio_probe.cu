#include "gpu/gpunetio_probe.hh"

#include <cuda_runtime.h>

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
      return opcode == MLX5_CQE_REQ_ERR ? -EIO : 0;
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
__global__ void gpunetio_read_probe_kernel(GpuNetioReadProbeParams params) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (params.remote_region_count == 0 || params.qp_array == nullptr ||
      params.qp_index >= params.qp_count ||
      params.remote_region >= params.remote_region_count ||
      params.qp_array[params.qp_index] == nullptr) {
    *params.status_code = -EINVAL;
    return;
  }
  const GpuNetioRemoteMemoryRegion& region = params.remote_regions[params.remote_region];
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

}  // namespace

void launch_gpunetio_read_probe(
    cudaStream_t stream, const GpuNetioReadProbeParams& params) {
  gpunetio_read_probe_kernel<<<1, 1, 0, stream>>>(params);
}

}  // namespace gpu
