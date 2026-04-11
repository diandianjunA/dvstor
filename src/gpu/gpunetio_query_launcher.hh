#pragma once

#include <cstddef>
#include <cstdint>

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace gpu {

inline constexpr uint32_t kGpuNetioDebugValueCount = 16;
inline constexpr uint64_t kGpuNetioNodeLockMask = 0x1ULL;

struct GpuNetioRemoteMemoryRegion {
  uint64_t address;
  uint32_t rkey;
  uint32_t reserved;
};

struct GpuNetioExactSearchParams {
  const float* query;
  uint32_t dim;
  uint32_t beam_width;
  uint32_t top_k;
  uint32_t max_results;
  uint32_t max_visited;
  uint32_t max_degree;
  uint32_t node_size;
  uint32_t offset_id;
  uint32_t offset_edge_count;
  uint32_t offset_vector;
  uint32_t offset_neighbors;
  uint32_t max_rdma_reads;
  uint32_t local_mkey;
  uint64_t local_iova_base;
  const GpuNetioRemoteMemoryRegion* remote_regions;
  uint32_t remote_region_count;
  void* const* qp_array;
  uint64_t* beam_ptrs;
  float* beam_dists;
  uint32_t* beam_expanded;
  uint64_t* visited_ptrs;
  uint32_t* result_ids;
  uint32_t* result_count;
  int* status_code;
  int* rdma_status_code;
  uint64_t* debug_values;
  unsigned char* node_a;
  unsigned char* node_b;
  uint64_t* medoid_ptr;
  unsigned char* dump_ptr;
};

void launch_gpunetio_exact_search(cudaStream_t stream, const GpuNetioExactSearchParams& params);

}  // namespace gpu
