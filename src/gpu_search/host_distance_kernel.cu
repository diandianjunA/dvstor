#include "gpu_search/host_distance_kernel.hh"

#include <cfloat>
#include <cstdint>

namespace gpu_search::host_distance {
namespace {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using f32 = float;
using byte_t = std::uint8_t;

inline constexpr f32 kMaximumSquaredL2 = 0x1.fffffcp+127f;

__device__ __forceinline__ bool finite_f32(f32 value) {
  return (__float_as_uint(value) & 0x7f800000u) != 0x7f800000u;
}

__device__ __forceinline__ f32 saturate_squared_l2_device(double value) {
  constexpr double maximum = static_cast<double>(kMaximumSquaredL2);
  if (!(value < maximum)) return kMaximumSquaredL2;
  return value <= 0.0 ? 0.0f : static_cast<f32>(value);
}

__global__ void pq_kernel(const u8* resident_codes,
                          const u32* resident_ordinals,
                          const u8* dynamic_codes,
                          u32 count,
                          u32 code_bytes,
                          const f32* distance_table,
                          f32* distances) {
  const u32 candidate = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate >= count) return;
  const u32 ordinal = resident_ordinals[candidate];
  const u8* code = ordinal == UINT32_MAX
    ? dynamic_codes + static_cast<size_t>(candidate) * code_bytes
    : resident_codes + static_cast<size_t>(ordinal) * code_bytes;
  f32 distance = 0.0f;
  for (u32 subquantizer = 0; subquantizer < code_bytes; ++subquantizer) {
    distance += distance_table[
      static_cast<size_t>(subquantizer) * 256u + code[subquantizer]];
  }
  distances[candidate] =
    finite_f32(distance) && distance < FLT_MAX
      ? distance : kMaximumSquaredL2;
}

__device__ __forceinline__ f32 storage_component(
    const byte_t* vector, u32 dtype, u32 dimension) {
  if (dtype == 0u) {
    return reinterpret_cast<const f32*>(vector)[dimension];
  }
  if (dtype == 1u) {
    return reinterpret_cast<const u8*>(vector)[dimension];
  }
  return reinterpret_cast<const std::int8_t*>(vector)[dimension];
}

__global__ void exact_kernel(const f32* query,
                             const byte_t* records,
                             u32 count,
                             u32 dim,
                             u32 dtype,
                             u32 record_stride,
                             u32 vector_offset,
                             f32* distances) {
  const u32 record_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (record_index >= count) return;
  const byte_t* vector = records +
    static_cast<size_t>(record_index) * record_stride + vector_offset;
  f32 distance = 0.0f;
  for (u32 dimension = 0; dimension < dim; ++dimension) {
    const f32 difference =
      query[dimension] - storage_component(vector, dtype, dimension);
    distance = fmaf(difference, difference, distance);
  }
  if (finite_f32(distance) && distance < FLT_MAX) {
    distances[record_index] = distance;
    return;
  }
  double wide_distance = 0.0;
  for (u32 dimension = 0; dimension < dim; ++dimension) {
    const double difference = static_cast<double>(query[dimension]) -
      static_cast<double>(storage_component(vector, dtype, dimension));
    wide_distance = fma(difference, difference, wide_distance);
  }
  distances[record_index] = saturate_squared_l2_device(wide_distance);
}

constexpr u32 kThreads = 128;

}  // namespace

void launch_pq(cudaStream_t stream,
               const u8* resident_codes,
               const u32* resident_ordinals,
               const u8* dynamic_codes,
               u32 count,
               u32 code_bytes,
               const f32* distance_table,
               f32* distances) {
  if (count == 0) return;
  pq_kernel<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
    resident_codes, resident_ordinals, dynamic_codes, count, code_bytes,
    distance_table, distances);
}

void launch_exact(cudaStream_t stream,
                  const f32* query,
                  const byte_t* records,
                  u32 count,
                  u32 dim,
                  VectorDType vector_dtype,
                  u32 record_stride,
                  u32 vector_offset,
                  f32* distances) {
  if (count == 0) return;
  exact_kernel<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
    query, records, count, dim, static_cast<u32>(vector_dtype),
    record_stride, vector_offset, distances);
}

}  // namespace gpu_search::host_distance
