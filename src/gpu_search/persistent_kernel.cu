#include <stdexcept>
#include <string>

#include "gpu_search/persistent_kernel.hh"
#include "gpu_search/persistent_kernel/runtime.cuh"

namespace gpu_search {

using namespace persistent_kernel_detail;

namespace {

template <u32 Threads, bool EnableAsfe, u32 PqSubquantizers>
const void* persistent_search_kernel_address() {
  return reinterpret_cast<const void*>(
    persistent_search_kernel<Threads, EnableAsfe, PqSubquantizers>);
}

template <u32 Threads, bool EnableAsfe>
const void* select_persistent_search_kernel(u32 pq_subquantizers) {
  switch (persistent_pq_kernel_specialization(pq_subquantizers)) {
    case kPersistentPq20Subquantizers:
      return persistent_search_kernel_address<
        Threads, EnableAsfe, kPersistentPq20Subquantizers>();
    case kPersistentPq25Subquantizers:
      return persistent_search_kernel_address<
        Threads, EnableAsfe, kPersistentPq25Subquantizers>();
    case kPersistentPq32Subquantizers:
      return persistent_search_kernel_address<
        Threads, EnableAsfe, kPersistentPq32Subquantizers>();
    default:
      return persistent_search_kernel_address<
        Threads, EnableAsfe, kPersistentRuntimePqSubquantizers>();
  }
}

template <u32 Threads, bool EnableAsfe>
void launch_persistent_search_kernel(cudaStream_t stream,
                                     const PersistentKernelParams& params,
                                     u32 blocks) {
  switch (persistent_pq_kernel_specialization(params.pq_subquantizers)) {
    case kPersistentPq20Subquantizers:
      persistent_search_kernel<Threads, EnableAsfe,
                               kPersistentPq20Subquantizers>
        <<<blocks, Threads, 0, stream>>>(params);
      return;
    case kPersistentPq25Subquantizers:
      persistent_search_kernel<Threads, EnableAsfe,
                               kPersistentPq25Subquantizers>
        <<<blocks, Threads, 0, stream>>>(params);
      return;
    case kPersistentPq32Subquantizers:
      persistent_search_kernel<Threads, EnableAsfe,
                               kPersistentPq32Subquantizers>
        <<<blocks, Threads, 0, stream>>>(params);
      return;
    default:
      persistent_search_kernel<Threads, EnableAsfe,
                               kPersistentRuntimePqSubquantizers>
        <<<blocks, Threads, 0, stream>>>(params);
      return;
  }
}

}  // namespace

PersistentKernelOccupancy inspect_persistent_search_kernel(u32 threads,
                                                           bool enable_asfe,
                                                           u32 pq_subquantizers) {
  if (threads != 128 && threads != 256) {
    throw std::invalid_argument(
      "persistent search kernel supports only 128 or 256 threads");
  }
  cudaFuncAttributes attributes{};
  const void* kernel = nullptr;
  if (threads == 128) {
    kernel =
      enable_asfe
        ? select_persistent_search_kernel<128, true>(pq_subquantizers)
        : select_persistent_search_kernel<128, false>(pq_subquantizers);
  } else {
    kernel =
      enable_asfe
        ? select_persistent_search_kernel<256, true>(pq_subquantizers)
        : select_persistent_search_kernel<256, false>(pq_subquantizers);
  }
  cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string("cudaFuncGetAttributes(persistent search): ") +
      cudaGetErrorString(status));
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_blocks, kernel, static_cast<int>(threads), 0);
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor(persistent search): ") +
      cudaGetErrorString(status));
  }
  const bool viable = active_blocks > 0 &&
    threads <= static_cast<u32>(attributes.maxThreadsPerBlock);
  return PersistentKernelOccupancy{
    .active_blocks_per_sm = viable ? static_cast<u32>(active_blocks) : 0u,
    .registers_per_thread = static_cast<u32>(attributes.numRegs),
    .static_shared_bytes = attributes.sharedSizeBytes,
    .local_bytes_per_thread = attributes.localSizeBytes,
    .max_threads_per_block = static_cast<u32>(attributes.maxThreadsPerBlock),
  };
}

void launch_persistent_search(cudaStream_t stream,
                              const PersistentKernelParams& params, u32 blocks,
                              u32 threads, bool decoupled_search_progression) {
  if (decoupled_search_progression &&
      params.issue_width < params.commit_width) {
    throw std::invalid_argument(
      "exact-frontier progression requires issue width >= commit width");
  }
  if (threads == 128) {
    if (decoupled_search_progression) {
      launch_persistent_search_kernel<128, true>(stream, params, blocks);
    } else {
      launch_persistent_search_kernel<128, false>(stream, params, blocks);
    }
  } else if (threads == 256) {
    if (decoupled_search_progression) {
      launch_persistent_search_kernel<256, true>(stream, params, blocks);
    } else {
      launch_persistent_search_kernel<256, false>(stream, params, blocks);
    }
  } else {
    throw std::invalid_argument(
      "persistent search kernel supports only 128 or 256 threads");
  }
}

void launch_direct_read_owners(cudaStream_t stream,
                               const PersistentKernelParams& params,
                               u32 queue_count, u32 threads) {
  const u32 warps_per_block = max(1u, threads / 32);
  const u32 blocks = (queue_count + warps_per_block - 1) / warps_per_block;
  direct_read_owner_kernel<<<blocks, threads, 0, stream>>>(params, queue_count);
}

void launch_gpunetio_owner_read_probe(cudaStream_t stream,
                                      const PersistentKernelParams& params,
                                      u32* request_shards, u64* remote_offsets,
                                      u64* local_iova_offsets, u8* destinations,
                                      u32 destination_stride, i32* statuses,
                                      u32* completed, u32* phases,
                                      u32 queue_count) {
  constexpr u32 threads = 128;
  const u32 blocks = (queue_count + threads - 1) / threads;
  gpunetio_owner_read_probe_kernel<<<blocks, threads, 0, stream>>>(
    params, request_shards, remote_offsets, local_iova_offsets, destinations,
    destination_stride, statuses, completed, phases, queue_count);
}

void launch_gpunetio_locked_read_probe(cudaStream_t stream,
                                       const PersistentKernelParams& params,
                                       u8* destinations, u32 destination_stride,
                                       i32* statuses, u32* completed,
                                       u32 blocks, u32 iterations) {
  gpunetio_locked_read_probe_kernel<<<blocks, 128, 0, stream>>>(
    params, destinations, destination_stride, statuses, completed, iterations);
}

void launch_gpunetio_batched_read_probe(cudaStream_t stream,
                                        const PersistentKernelParams& params,
                                        u8* destinations,
                                        u32 destination_stride, i32* statuses,
                                        u32* completed, u32 blocks,
                                        u32 batch_size) {
  gpunetio_batched_read_probe_kernel<<<blocks, 128, 0, stream>>>(
    params, destinations, destination_stride, statuses, completed, batch_size);
}

}  // namespace gpu_search
