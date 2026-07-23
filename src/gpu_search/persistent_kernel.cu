#include "gpu_search/persistent_kernel.hh"
#include "gpu_search/persistent_kernel/runtime.cuh"

#include <stdexcept>
#include <string>

namespace gpu_search {

using namespace persistent_kernel_detail;

PersistentKernelOccupancy inspect_persistent_search_kernel(u32 threads) {
  if (threads != 128 && threads != 256) {
    throw std::invalid_argument(
      "persistent search kernel supports only 128 or 256 threads");
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
    &attributes, persistent_search_kernel);
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string("cudaFuncGetAttributes(persistent search): ") +
      cudaGetErrorString(status));
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_blocks, persistent_search_kernel, static_cast<int>(threads), 0);
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string("cudaOccupancyMaxActiveBlocksPerMultiprocessor(persistent search): ") +
      cudaGetErrorString(status));
  }
  if (active_blocks <= 0 ||
      threads > static_cast<u32>(attributes.maxThreadsPerBlock)) {
    throw std::runtime_error(
      "persistent search kernel has zero occupancy for the requested CTA size");
  }
  return PersistentKernelOccupancy{
    .active_blocks_per_sm = static_cast<u32>(active_blocks),
    .registers_per_thread = static_cast<u32>(attributes.numRegs),
    .static_shared_bytes = attributes.sharedSizeBytes,
    .max_threads_per_block =
      static_cast<u32>(attributes.maxThreadsPerBlock),
  };
}

void launch_persistent_search(cudaStream_t stream, const PersistentKernelParams& params,
                              u32 blocks, u32 threads) {
  persistent_search_kernel<<<blocks, threads, 0, stream>>>(params);
}

void launch_direct_read_owners(cudaStream_t stream,
                               const PersistentKernelParams& params,
                               u32 queue_count, u32 threads) {
  const u32 warps_per_block = max(1u, threads / 32);
  const u32 blocks = (queue_count + warps_per_block - 1) / warps_per_block;
  direct_read_owner_kernel<<<blocks, threads, 0, stream>>>(params, queue_count);
}

void launch_gpunetio_owner_read_probe(
    cudaStream_t stream, const PersistentKernelParams& params,
    u32* request_shards, u64* remote_offsets, u64* local_iova_offsets,
    u8* destinations, u32 destination_stride, i32* statuses,
    u32* completed, u32* phases, u32 queue_count) {
  constexpr u32 threads = 128;
  const u32 blocks = (queue_count + threads - 1) / threads;
  gpunetio_owner_read_probe_kernel<<<blocks, threads, 0, stream>>>(
    params, request_shards, remote_offsets, local_iova_offsets,
    destinations, destination_stride, statuses, completed, phases, queue_count);
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
                                        u8* destinations, u32 destination_stride,
                                        i32* statuses, u32* completed,
                                        u32 blocks, u32 batch_size) {
  gpunetio_batched_read_probe_kernel<<<blocks, 128, 0, stream>>>(
    params, destinations, destination_stride, statuses, completed, batch_size);
}

}  // namespace gpu_search
