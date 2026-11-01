#include <stdexcept>
#include <string>

#include "gpu_search/persistent_kernel.hh"
#include "gpu_search/persistent_kernel/runtime.cuh"

namespace gpu_search {

using namespace persistent_kernel_detail;

namespace {

template <u32 Threads, bool EnableAsfe, u32 PqSubquantizers>
const void* persistent_search_kernel_address() {
  const void* kernel = reinterpret_cast<const void*>(
    persistent_search_kernel<Threads, EnableAsfe, PqSubquantizers>);
  // The LUT tile scorer rides on dynamic shared memory.  Static shared
  // (48,544 B) plus the tile exceeds the default 48-KiB per-block shared
  // limit, so every instantiation opts into the extended limit once before
  // its first occupancy query or launch.
  static const bool configured = [kernel] {
    const cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kPersistentPqLutTileBytes));
    if (status != cudaSuccess) {
      throw std::runtime_error(
        std::string("cudaFuncSetAttribute(persistent search LUT tile): ") +
        cudaGetErrorString(status));
    }
    return true;
  }();
  (void)configured;
  return kernel;
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
    case kPersistentPq64Subquantizers:
      return persistent_search_kernel_address<
        Threads, EnableAsfe, kPersistentPq64Subquantizers>();
    case kPersistentPq128Subquantizers:
      return persistent_search_kernel_address<
        Threads, EnableAsfe, kPersistentPq128Subquantizers>();
    case kPersistentPq256Subquantizers:
      return persistent_search_kernel_address<
        Threads, EnableAsfe, kPersistentPq256Subquantizers>();
    default:
      return persistent_search_kernel_address<
        Threads, EnableAsfe, kPersistentRuntimePqSubquantizers>();
  }
}

template <u32 Threads, bool EnableAsfe>
void launch_persistent_search_kernel(cudaStream_t stream,
                                     const PersistentKernelParams& params,
                                     u32 blocks) {
  // The static-code PQ scorer stages its LUT tiles in dynamic shared memory
  // (score_static_candidates_lut_tiled); every specialization carries the
  // same tile capacity.  Launching through the selected pointer reuses the
  // one-time shared-memory attribute configuration above.
  using KernelEntry = void (*)(PersistentKernelParams);
  const auto kernel = reinterpret_cast<KernelEntry>(const_cast<void*>(
    select_persistent_search_kernel<Threads, EnableAsfe>(
      params.pq_subquantizers)));
  kernel<<<blocks, Threads, kPersistentPqLutTileBytes, stream>>>(params);
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
#if DVSTOR_PERSISTENT_WIDE_SORT
    kernel =
      enable_asfe
        ? select_persistent_search_kernel<256, true>(pq_subquantizers)
        : select_persistent_search_kernel<256, false>(pq_subquantizers);
#else
    // Slim builds compile out the 256-thread sort paths; report the
    // configuration as non-viable so the grid planner always picks 128.
    return PersistentKernelOccupancy{};
#endif
  }
  cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string("cudaFuncGetAttributes(persistent search): ") +
      cudaGetErrorString(status));
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_blocks, kernel, static_cast<int>(threads),
    kPersistentPqLutTileBytes);
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
#if DVSTOR_PERSISTENT_WIDE_SORT
    if (decoupled_search_progression) {
      launch_persistent_search_kernel<256, true>(stream, params, blocks);
    } else {
      launch_persistent_search_kernel<256, false>(stream, params, blocks);
    }
#else
    throw std::invalid_argument(
      "persistent search kernel was built without 256-thread support "
      "(DVSTOR_PERSISTENT_WIDE_SORT=OFF)");
#endif
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
