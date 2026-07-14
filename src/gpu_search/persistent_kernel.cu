#include "gpu_search/persistent_kernel.hh"
#include "gpu_search/persistent_kernel/runtime.cuh"

namespace gpu_search {

using namespace persistent_kernel_detail;

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

void launch_gather_anchor_codes(cudaStream_t stream, const u8* base_codes,
                                const u32* anchor_handles, u8* anchor_codes,
                                u32 anchor_count, u32 code_bytes,
                                u32 node_count) {
  const u64 bytes = static_cast<u64>(anchor_count) * code_bytes;
  if (bytes == 0) return;
  constexpr u32 threads = 256;
  const u32 blocks = static_cast<u32>((bytes + threads - 1) / threads);
  gather_anchor_codes_kernel<<<blocks, threads, 0, stream>>>(
    base_codes, anchor_handles, anchor_codes, anchor_count, code_bytes, node_count);
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
