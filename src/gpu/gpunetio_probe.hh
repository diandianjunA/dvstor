#pragma once

#include <cstddef>
#include <cstdint>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace gpu {

inline constexpr uint32_t kGpuNetioProbeDebugValueCount = 40;

struct GpuNetioRemoteMemoryRegion {
  uint64_t address;
  uint32_t rkey;
  uint32_t reserved;
};

struct GpuNetioReadProbeParams {
  uint32_t local_mkey;
  uint64_t local_iova_base;
  const GpuNetioRemoteMemoryRegion* remote_regions;
  uint32_t remote_region_count;
  void* const* qp_array;
  uint32_t qp_count;
  uint32_t qp_index;
  uint32_t remote_region;
  unsigned char* destination;
  unsigned char* dump_ptr;
  int* status_code;
  uint64_t* debug_values;
};

void launch_gpunetio_read_probe(
  cudaStream_t stream, const GpuNetioReadProbeParams& params);

}  // namespace gpu
