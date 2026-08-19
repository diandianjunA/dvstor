#pragma once

#include <cstdint>

enum class VectorDType : std::uint32_t;
struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace gpu_search::host_distance {

// Finite-lifetime kernels used by the CPU-orchestrated baseline.  They are
// deliberately ordinary stream launches: selecting this backend never starts
// the persistent query kernel and never gives a GPU thread ownership of an
// RDMA queue pair.
void launch_pq(cudaStream_t stream,
               const std::uint8_t* resident_codes,
               const std::uint32_t* resident_ordinals,
               const std::uint8_t* dynamic_codes,
               std::uint32_t count,
               std::uint32_t code_bytes,
               const float* distance_table,
               float* distances);

void launch_exact(cudaStream_t stream,
                  const float* query,
                  const std::uint8_t* records,
                  std::uint32_t count,
                  std::uint32_t dim,
                  VectorDType vector_dtype,
                  std::uint32_t record_stride,
                  std::uint32_t vector_offset,
                  float* distances);

}  // namespace gpu_search::host_distance
