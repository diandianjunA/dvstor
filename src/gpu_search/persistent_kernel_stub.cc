// Linker stubs for the persistent-kernel entry points when the runtime is
// built without DOCA GPUNetIO (DVSTOR_HAVE_GPUNETIO off). The persistent
// engine constructor rejects that configuration before any of these can run,
// so each stub simply throws if it is ever reached.
#include "gpu_search/persistent_kernel.hh"

#include <stdexcept>

namespace gpu_search {
namespace {

[[noreturn]] void throw_no_gpunetio(const char* entry) {
  throw std::logic_error(std::string(entry) +
                         ": persistent kernel unavailable in a build without "
                         "DOCA GPUNetIO support");
}

}  // namespace

PersistentKernelOccupancy inspect_persistent_search_kernel(
  u32 /*threads*/, bool /*enable_asfe*/, u32 /*pq_subquantizers*/) {
  throw_no_gpunetio("inspect_persistent_search_kernel");
}

void launch_persistent_search(cudaStream_t /*stream*/,
                              const PersistentKernelParams& /*params*/,
                              u32 /*blocks*/, u32 /*threads*/,
                              bool /*decoupled_search_progression*/) {
  throw_no_gpunetio("launch_persistent_search");
}

void launch_direct_read_owners(cudaStream_t /*stream*/,
                               const PersistentKernelParams& /*params*/,
                               u32 /*queue_count*/, u32 /*threads*/) {
  throw_no_gpunetio("launch_direct_read_owners");
}

void launch_gpunetio_owner_read_probe(
  cudaStream_t /*stream*/, const PersistentKernelParams& /*params*/,
  u32* /*request_shards*/, u64* /*remote_offsets*/,
  u64* /*local_iova_offsets*/, u8* /*destinations*/,
  u32 /*destination_stride*/, i32* /*statuses*/, u32* /*completed*/,
  u32* /*phases*/, u32 /*queue_count*/) {
  throw_no_gpunetio("launch_gpunetio_owner_read_probe");
}

void launch_gpunetio_locked_read_probe(
  cudaStream_t /*stream*/, const PersistentKernelParams& /*params*/,
  u8* /*destinations*/, u32 /*destination_stride*/, i32* /*statuses*/,
  u32* /*completed*/, u32 /*blocks*/, u32 /*iterations*/) {
  throw_no_gpunetio("launch_gpunetio_locked_read_probe");
}

void launch_gpunetio_batched_read_probe(
  cudaStream_t /*stream*/, const PersistentKernelParams& /*params*/,
  u8* /*destinations*/, u32 /*destination_stride*/, i32* /*statuses*/,
  u32* /*completed*/, u32 /*blocks*/, u32 /*batch_size*/) {
  throw_no_gpunetio("launch_gpunetio_batched_read_probe");
}

}  // namespace gpu_search
