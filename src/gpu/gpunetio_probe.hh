#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace gpu {

inline constexpr uint32_t kGpuNetioProbeDebugValueCount = 40;

struct GpuNetioRemoteMemoryRegion {
  uint64_t address;
  uint32_t rkey;
  uint32_t reserved;
  uint64_t bytes;
};

static_assert(sizeof(GpuNetioRemoteMemoryRegion) == 24);
static_assert(std::is_standard_layout_v<GpuNetioRemoteMemoryRegion>);
static_assert(std::is_trivially_copyable_v<GpuNetioRemoteMemoryRegion>);
static_assert(offsetof(GpuNetioRemoteMemoryRegion, address) == 0);
static_assert(offsetof(GpuNetioRemoteMemoryRegion, rkey) == 8);
static_assert(offsetof(GpuNetioRemoteMemoryRegion, reserved) == 12);
static_assert(offsetof(GpuNetioRemoteMemoryRegion, bytes) == 16);

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

static_assert(std::is_standard_layout_v<GpuNetioReadProbeParams>);
static_assert(std::is_trivially_copyable_v<GpuNetioReadProbeParams>);

void launch_gpunetio_read_probe(
  cudaStream_t stream, const GpuNetioReadProbeParams& params);

// A read-only transport microbenchmark. Each active warp owns one QP and
// repeatedly submits a fixed-size batch of one-sided RDMA READ WQEs. A
// non-zero second_stage_bytes models a dependent header/body protocol: the
// body batch is submitted only after the header batch completion is visible.
// This is deliberately separate from the persistent query kernel.
struct GpuNetioPayloadProbeParams {
  uint32_t local_mkey;
  uint64_t local_iova_base;
  const GpuNetioRemoteMemoryRegion* remote_regions;
  uint32_t remote_region_count;
  void* const* qp_array;
  uint32_t qp_count;
  uint32_t active_qps;
  unsigned char* destination;
  uint32_t destination_stride;
  uint32_t remote_record_stride;
  uint64_t remote_span_bytes;
  unsigned char* dump_ptr;
  uint32_t first_stage_bytes;
  uint32_t second_stage_bytes;
  uint32_t batch_reads;
  uint32_t warmup_batches;
  uint32_t measured_batches;
  uint64_t timeout_ns;
  int* status_codes;
  uint64_t* completed_reads;
  uint32_t* dump_wqe_flags;
  uint64_t* batch_latency_ns;
};

static_assert(std::is_standard_layout_v<GpuNetioPayloadProbeParams>);
static_assert(std::is_trivially_copyable_v<GpuNetioPayloadProbeParams>);

void launch_gpunetio_payload_probe(
  cudaStream_t stream, const GpuNetioPayloadProbeParams& params);

}  // namespace gpu
