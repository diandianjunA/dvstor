#include "gpu/gpunetio_query_launcher.hh"

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

__device__ inline uint32_t remote_region_index(const uint64_t raw_remote_ptr) {
  return static_cast<uint32_t>(raw_remote_ptr >> 48);
}

__device__ inline uint64_t remote_region_offset(const uint64_t raw_remote_ptr) {
  return (raw_remote_ptr << 16) >> 16;
}

__device__ inline bool visited_contains(const uint64_t* visited_ptrs, uint32_t visited_count, uint64_t value) {
  for (uint32_t i = 0; i < visited_count; ++i) {
    if (visited_ptrs[i] == value) {
      return true;
    }
  }
  return false;
}

__device__ inline bool node_is_locked(const unsigned char* node) {
  return (*reinterpret_cast<const uint64_t*>(node) & kGpuNetioNodeLockMask) != 0;
}

__device__ inline float l2_distance_squared(const float* query, const float* candidate, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t d = 0; d < dim; ++d) {
    const float diff = query[d] - candidate[d];
    sum += diff * diff;
  }
  return sum;
}

__device__ inline void insert_into_beam(uint64_t* beam_ptrs,
                                        float* beam_dists,
                                        uint32_t* beam_expanded,
                                        uint32_t& beam_count,
                                        const uint32_t beam_width,
                                        const uint64_t candidate_ptr,
                                        const float candidate_dist) {
  if (beam_count < beam_width) {
    beam_ptrs[beam_count] = candidate_ptr;
    beam_dists[beam_count] = candidate_dist;
    beam_expanded[beam_count] = 0;
    ++beam_count;
    return;
  }

  uint32_t worst_idx = 0;
  float worst_dist = beam_dists[0];
  for (uint32_t i = 1; i < beam_count; ++i) {
    if (beam_dists[i] > worst_dist) {
      worst_idx = i;
      worst_dist = beam_dists[i];
    }
  }

  if (candidate_dist >= worst_dist) {
    return;
  }

  beam_ptrs[worst_idx] = candidate_ptr;
  beam_dists[worst_idx] = candidate_dist;
  beam_expanded[worst_idx] = 0;
}

__device__ inline void set_rdma_debug(uint64_t* debug_values,
                                      const uint64_t stage,
                                      const uint64_t raw_ptr,
                                      const uint64_t remote_addr,
                                      const void* local_addr,
                                      const uint64_t local_iova_base) {
  debug_values[10] = stage;
  debug_values[11] = raw_ptr;
  debug_values[12] = remote_addr;
  debug_values[13] = reinterpret_cast<uint64_t>(local_addr) - local_iova_base;
}

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
      doca_gpu_dev_verbs_load_relaxed<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(&cq->cqe_ci);
    opown = doca_gpu_dev_verbs_load_relaxed_sys_global(reinterpret_cast<uint8_t*>(&cqe64->op_own));
    if (!((curr_cons_index <= ticket) && ((opown & MLX5_CQE_OWNER_MASK) ^ !!(ticket & cqe_num)))) {
      const uint8_t opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;
      const int status = (opcode == MLX5_CQE_REQ_ERR) * -EIO;
      if (status == 0) {
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
        doca_gpu_dev_verbs_atomic_max<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
          &cq->cqe_ci, ticket + 1);
        const uint32_t cq_ci = static_cast<uint32_t>((ticket + 1) & DOCA_GPUNETIO_VERBS_CQE_CI_MASK);
        asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;"
                     :
                     : "l"(cq->dbrec), "r"(doca_gpu_dev_verbs_bswap32(cq_ci)));
      }
      return status;
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

__device__ inline int gpudirect_get(void* qp_handle,
                                    const GpuNetioRemoteMemoryRegion* regions,
                                    const uint32_t region_count,
                                    const uint32_t local_mkey,
                                    const uint64_t local_iova_base,
                                    const uint32_t region_idx,
                                    const uint64_t remote_addr,
                                    void* local_addr,
                                    void* dump_addr,
                                    const size_t size,
                                    uint64_t* cqe_debug = nullptr) {
  if (qp_handle == nullptr || region_idx >= region_count) {
    return -1;
  }

  auto* qp = reinterpret_cast<struct doca_gpu_dev_verbs_qp*>(qp_handle);
  doca_gpu_dev_verbs_ticket_t ticket = 0;
  struct doca_gpu_dev_verbs_addr raddr{
    .addr = remote_addr,
    .key = regions[region_idx].rkey,
  };
  const uint64_t local_cuda_addr = reinterpret_cast<uint64_t>(local_addr);
  struct doca_gpu_dev_verbs_addr laddr{
    .addr = local_cuda_addr - local_iova_base,
    .key = local_mkey,
  };
  struct doca_gpu_dev_verbs_addr daddr{
    .addr = reinterpret_cast<uint64_t>(dump_addr) - local_iova_base,
    .key = local_mkey,
  };

  doca_gpu_dev_verbs_get<DOCA_GPUNETIO_VERBS_NODUMP,
                         DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE,
                         DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(qp, raddr, laddr, size, daddr, &ticket);
  auto* cq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
  const int status = poll_cq_at_with_timeout(cq, ticket, cqe_debug);
  if (status != 0 && status != kPollTimeoutStatus && cqe_debug != nullptr) {
    auto* cqe_base = reinterpret_cast<struct mlx5_cqe64*>(cq->cqe_daddr);
    auto* err_cqe = reinterpret_cast<struct mlx5_err_cqe_ex*>(&cqe_base[ticket & (cq->cqe_num - 1)]);
    cqe_debug[0] = (static_cast<uint64_t>(err_cqe->syndrome) << 0) |
                   (static_cast<uint64_t>(err_cqe->vendor_err_synd) << 8) |
                   (static_cast<uint64_t>(err_cqe->hw_err_synd) << 16) |
                   (static_cast<uint64_t>(err_cqe->hw_synd_type) << 24) |
                   (static_cast<uint64_t>(err_cqe->op_own) << 32);
    cqe_debug[1] = (static_cast<uint64_t>(err_cqe->wqe_counter) << 32) | err_cqe->s_wqe_opcode_qpn;
  }
  return status;
}

__global__ void gpunetio_exact_search_kernel(GpuNetioExactSearchParams params) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  *params.status_code = 0;
  *params.rdma_status_code = 0;
  *params.result_count = 0;
  for (uint32_t i = 0; i < kGpuNetioDebugValueCount; ++i) {
    params.debug_values[i] = 0;
  }

  if (params.remote_region_count == 0 || params.beam_width == 0 || params.max_results == 0) {
    *params.status_code = -10;
    return;
  }

  uint32_t rdma_reads = 0;
  params.debug_values[0] = params.remote_regions[0].address + 8;
  params.debug_values[1] = params.remote_regions[0].rkey;
  params.debug_values[2] = reinterpret_cast<uint64_t>(params.medoid_ptr) - params.local_iova_base;
  params.debug_values[3] = params.local_mkey;
  params.debug_values[5] = params.remote_regions[0].reserved;
  set_rdma_debug(params.debug_values,
                 1,
                 0,
                 params.remote_regions[0].address + 8,
                 params.medoid_ptr,
                 params.local_iova_base);
  const int medoid_read_status = gpudirect_get(params.qp_array[0],
                                               params.remote_regions,
                                               params.remote_region_count,
                                               params.local_mkey,
                                               params.local_iova_base,
                                               0,
                                               params.remote_regions[0].address + 8,
                                               params.medoid_ptr,
                                               params.dump_ptr,
                                               sizeof(uint64_t),
                                               &params.debug_values[6]);
  *params.rdma_status_code = medoid_read_status;
  if (medoid_read_status != 0) {
    *params.status_code = -11;
    return;
  }
  ++rdma_reads;

  const uint64_t medoid_raw = *params.medoid_ptr;
  params.debug_values[4] = medoid_raw;
  if (medoid_raw == 0) {
    return;
  }

  uint32_t beam_count = 0;
  uint32_t visited_count = 0;
  params.visited_ptrs[visited_count++] = medoid_raw;

  const uint32_t medoid_region = remote_region_index(medoid_raw);
  const uint64_t medoid_offset = remote_region_offset(medoid_raw);
  if (medoid_region >= params.remote_region_count) {
    *params.status_code = -18;
    return;
  }
  set_rdma_debug(params.debug_values,
                 2,
                 medoid_raw,
                 params.remote_regions[medoid_region].address + medoid_offset,
                 params.node_a,
                 params.local_iova_base);
  const int medoid_node_status = gpudirect_get(params.qp_array[medoid_region],
                                               params.remote_regions,
                                               params.remote_region_count,
                                               params.local_mkey,
                                               params.local_iova_base,
                                               medoid_region,
                                               params.remote_regions[medoid_region].address + medoid_offset,
                                               params.node_a,
                                               params.dump_ptr,
                                               params.node_size,
                                               &params.debug_values[6]);
  *params.rdma_status_code = medoid_node_status;
  if (medoid_node_status != 0) {
    *params.status_code = -12;
    return;
  }
  ++rdma_reads;
  params.debug_values[14] = *reinterpret_cast<const uint32_t*>(params.node_a + params.offset_id);
  params.debug_values[15] = *(params.node_a + params.offset_edge_count);
  if (node_is_locked(params.node_a)) {
    return;
  }

  const float* medoid_vector = reinterpret_cast<const float*>(params.node_a + params.offset_vector);
  params.beam_ptrs[0] = medoid_raw;
  params.beam_dists[0] = l2_distance_squared(params.query, medoid_vector, params.dim);
  params.beam_expanded[0] = 0;
  beam_count = 1;

  bool rdma_budget_exhausted = false;
  while (!rdma_budget_exhausted) {
    int best_idx = -1;
    float best_dist = 0.0f;
    for (uint32_t i = 0; i < beam_count; ++i) {
      if (params.beam_expanded[i] != 0) {
        continue;
      }
      if (best_idx < 0 || params.beam_dists[i] < best_dist) {
        best_idx = static_cast<int>(i);
        best_dist = params.beam_dists[i];
      }
    }

    if (best_idx < 0) {
      break;
    }

    params.beam_expanded[best_idx] = 1;
    const uint64_t current_raw = params.beam_ptrs[best_idx];
    const uint32_t current_region = remote_region_index(current_raw);
    const uint64_t current_offset = remote_region_offset(current_raw);
    if (current_region >= params.remote_region_count) {
      *params.status_code = -19;
      return;
    }
    if (rdma_reads + params.top_k + 1 >= params.max_rdma_reads) {
      rdma_budget_exhausted = true;
      break;
    }
    set_rdma_debug(params.debug_values,
                   3,
                   current_raw,
                   params.remote_regions[current_region].address + current_offset,
                   params.node_a,
                   params.local_iova_base);
    const int current_node_status = gpudirect_get(params.qp_array[current_region],
                                                  params.remote_regions,
                                                  params.remote_region_count,
                                                  params.local_mkey,
                                                  params.local_iova_base,
                                                  current_region,
                                                  params.remote_regions[current_region].address + current_offset,
                                                  params.node_a,
                                                  params.dump_ptr,
                                                  params.node_size,
                                                  &params.debug_values[6]);
    *params.rdma_status_code = current_node_status;
    if (current_node_status != 0) {
      *params.status_code = -13;
      return;
    }
    ++rdma_reads;
    if (node_is_locked(params.node_a)) {
      continue;
    }

    const uint8_t edge_count = *(params.node_a + params.offset_edge_count);
    auto* neighbors = reinterpret_cast<const uint64_t*>(params.node_a + params.offset_neighbors);
    const uint32_t neighbor_count = edge_count > params.max_degree ? params.max_degree : edge_count;

    for (uint32_t n = 0; n < neighbor_count; ++n) {
      const uint64_t neighbor_raw = neighbors[n];
      if (neighbor_raw == 0 || visited_contains(params.visited_ptrs, visited_count, neighbor_raw)) {
        continue;
      }
      if (visited_count >= params.max_visited) {
        *params.status_code = -14;
        return;
      }

      params.visited_ptrs[visited_count++] = neighbor_raw;

      const uint32_t neighbor_region = remote_region_index(neighbor_raw);
      const uint64_t neighbor_offset = remote_region_offset(neighbor_raw);
      if (neighbor_region >= params.remote_region_count) {
        *params.status_code = -15;
        return;
      }
      if (rdma_reads + params.top_k + 1 >= params.max_rdma_reads) {
        rdma_budget_exhausted = true;
        break;
      }
      set_rdma_debug(params.debug_values,
                     4,
                     neighbor_raw,
                     params.remote_regions[neighbor_region].address + neighbor_offset,
                     params.node_b,
                     params.local_iova_base);
      const int neighbor_node_status = gpudirect_get(params.qp_array[neighbor_region],
                                                     params.remote_regions,
                                                     params.remote_region_count,
                                                     params.local_mkey,
                                                     params.local_iova_base,
                                                     neighbor_region,
                                                     params.remote_regions[neighbor_region].address + neighbor_offset,
                                                     params.node_b,
                                                     params.dump_ptr,
                                                     params.node_size,
                                                     &params.debug_values[6]);
      *params.rdma_status_code = neighbor_node_status;
      if (neighbor_node_status != 0) {
        *params.status_code = -16;
        return;
      }
      ++rdma_reads;
      if (node_is_locked(params.node_b)) {
        continue;
      }

      const float* neighbor_vector = reinterpret_cast<const float*>(params.node_b + params.offset_vector);
      const float dist = l2_distance_squared(params.query, neighbor_vector, params.dim);
      insert_into_beam(params.beam_ptrs,
                       params.beam_dists,
                       params.beam_expanded,
                       beam_count,
                       params.beam_width,
                       neighbor_raw,
                       dist);
    }
  }

  for (uint32_t i = 0; i < beam_count; ++i) {
    uint32_t best = i;
    for (uint32_t j = i + 1; j < beam_count; ++j) {
      if (params.beam_dists[j] < params.beam_dists[best]) {
        best = j;
      }
    }
    if (best != i) {
      const uint64_t ptr_tmp = params.beam_ptrs[i];
      params.beam_ptrs[i] = params.beam_ptrs[best];
      params.beam_ptrs[best] = ptr_tmp;

      const float dist_tmp = params.beam_dists[i];
      params.beam_dists[i] = params.beam_dists[best];
      params.beam_dists[best] = dist_tmp;

      const uint32_t expanded_tmp = params.beam_expanded[i];
      params.beam_expanded[i] = params.beam_expanded[best];
      params.beam_expanded[best] = expanded_tmp;
    }
  }

  const uint32_t result_count = params.top_k < beam_count ? params.top_k : beam_count;
  *params.result_count = result_count;

  for (uint32_t i = 0; i < result_count; ++i) {
    const uint64_t node_raw = params.beam_ptrs[i];
    const uint32_t node_region = remote_region_index(node_raw);
    const uint64_t node_offset = remote_region_offset(node_raw);
    if (node_region >= params.remote_region_count) {
      *params.status_code = -20;
      return;
    }
    if (rdma_reads >= params.max_rdma_reads) {
      *params.result_count = i;
      return;
    }
    set_rdma_debug(params.debug_values,
                   5,
                   node_raw,
                   params.remote_regions[node_region].address + node_offset,
                   params.node_b,
                   params.local_iova_base);
    const int result_node_status = gpudirect_get(params.qp_array[node_region],
                                                 params.remote_regions,
                                                 params.remote_region_count,
                                                 params.local_mkey,
                                                 params.local_iova_base,
                                                 node_region,
                                                 params.remote_regions[node_region].address + node_offset,
                                                 params.node_b,
                                                 params.dump_ptr,
                                                 params.node_size,
                                                 &params.debug_values[6]);
    *params.rdma_status_code = result_node_status;
    if (result_node_status != 0) {
      *params.status_code = -17;
      return;
    }
    ++rdma_reads;

    params.result_ids[i] = *reinterpret_cast<const uint32_t*>(params.node_b + params.offset_id);
  }
  params.debug_values[8] = rdma_reads;
}

}  // namespace

void launch_gpunetio_exact_search(cudaStream_t stream, const GpuNetioExactSearchParams& params) {
  gpunetio_exact_search_kernel<<<1, 1, 0, stream>>>(params);
}

}  // namespace gpu
