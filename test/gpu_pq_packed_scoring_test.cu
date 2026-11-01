#include <cuda_runtime.h>

#include <cfloat>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_search/persistent_kernel/candidate_scoring.cuh"

namespace {

using gpu_search::DeviceShardRegion;
using gpu_search::PersistentKernelParams;
using gpu_search::f32;
using gpu_search::u32;
using gpu_search::u64;
using gpu_search::u8;
using gpu_search::persistent_kernel_detail::approximate_entry;
using gpu_search::persistent_kernel_detail::score_static_candidates_lut_tiled;
using gpu_search::persistent_kernel_detail::static_ordinal_from_raw;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                          count * sizeof(T)), "cudaMalloc");
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() const { return data_; }
  size_t size() const { return count_; }

 private:
  T* data_{};
  size_t count_{};
};

__device__ f32 scalar_reference(const f32* query_lut, const u8* code,
                                u32 subquantizers) {
  f32 distance = 0.0f;
  for (u32 subquantizer = 0; subquantizer < subquantizers; ++subquantizer) {
    distance += query_lut[static_cast<size_t>(subquantizer) * 256u +
                          code[subquantizer]];
  }
  return distance;
}

template <u32 PqSubquantizers>
__global__ void compare_scoring_kernel(PersistentKernelParams params,
                                       const f32* query_lut,
                                       const u8* codes, u32 code_stride,
                                       u32 cases, u32* mismatches) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= cases) return;
  // Production codes are tightly packed. PQ25 therefore encounters all four
  // alignments, while the PQ20/PQ32 strides preserve cudaMalloc alignment.
  const u8* code = codes + static_cast<size_t>(index) * code_stride;
  const f32 expected =
    scalar_reference(query_lut, code, params.pq_subquantizers);
  const f32 actual =
    approximate_entry<PqSubquantizers>(params, query_lut, code);
  if (__float_as_uint(expected) != __float_as_uint(actual)) {
    atomicAdd(mismatches, 1u);
  }
}

std::vector<f32> make_query_lut(u32 subquantizers) {
  std::vector<f32> query_lut(static_cast<size_t>(subquantizers) * 256u);
  for (u32 subquantizer = 0; subquantizer < subquantizers; ++subquantizer) {
    for (u32 centroid = 0; centroid < 256; ++centroid) {
      // Binary fractions keep the equality assertion sensitive to addition
      // order without introducing host/device decimal conversion noise.
      query_lut[static_cast<size_t>(subquantizer) * 256u + centroid] =
        static_cast<f32>((subquantizer * 257u + centroid) & 0xffffu) /
        1024.0f;
    }
  }
  return query_lut;
}

template <u32 PqSubquantizers>
void run_case(u32 subquantizers) {
  constexpr u32 kCases = 4096;
  std::vector<f32> query_lut = make_query_lut(subquantizers);
  std::vector<u8> codes(
    static_cast<size_t>(kCases) * subquantizers +
    gpu_search::kPersistentMaxSubquantizers);
  for (u32 index = 0; index < codes.size(); ++index) {
    codes[index] = static_cast<u8>((index * 73u + 19u) & 0xffu);
  }

  DeviceBuffer<f32> device_lut(query_lut.size());
  DeviceBuffer<u8> device_codes(codes.size());
  DeviceBuffer<u32> device_mismatches(1);
  check_cuda(cudaMemcpy(device_lut.get(), query_lut.data(),
                        query_lut.size() * sizeof(f32),
                        cudaMemcpyHostToDevice), "cudaMemcpy LUT H2D");
  check_cuda(cudaMemcpy(device_codes.get(), codes.data(),
                        codes.size() * sizeof(u8), cudaMemcpyHostToDevice),
             "cudaMemcpy codes H2D");
  check_cuda(cudaMemset(device_mismatches.get(), 0, sizeof(u32)),
             "cudaMemset mismatches");

  PersistentKernelParams params{};
  params.pq_subquantizers = subquantizers;
  compare_scoring_kernel<PqSubquantizers>
    <<<(kCases + 127u) / 128u, 128>>>(
    params, device_lut.get(), device_codes.get(), subquantizers, kCases,
    device_mismatches.get());
  check_cuda(cudaGetLastError(), "compare_scoring_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "compare_scoring_kernel completion");

  u32 mismatches = 0;
  check_cuda(cudaMemcpy(&mismatches, device_mismatches.get(), sizeof(u32),
                        cudaMemcpyDeviceToHost), "cudaMemcpy mismatches D2H");
  if (mismatches != 0) {
    throw std::runtime_error(
      "packed PQ" + std::to_string(subquantizers) +
      " scoring changed scalar distance results for " +
      std::to_string(mismatches) + " cases");
  }
}

// One CTA scores the whole static wave through the shared-memory LUT tile
// path, mirroring the production approximate_handles_batch contract:
// distances[] starts at the FLT_MAX sentinel for every candidate.
template <u32 PqSubquantizers>
__global__ void tiled_scoring_kernel(PersistentKernelParams params,
                                     const f32* query_lut,
                                     const u64* handles, u32 count,
                                     f32* distances) {
  extern __shared__ f32 lut_tile[];
  for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
    distances[index] = FLT_MAX;
  }
  __syncthreads();
  score_static_candidates_lut_tiled<PqSubquantizers>(
    params, query_lut, handles, count, distances, lut_tile);
}

__global__ void compare_tiled_kernel(PersistentKernelParams params,
                                     const f32* query_lut,
                                     const u64* handles, u32 count,
                                     const f32* distances,
                                     u32* mismatches) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  f32 expected = FLT_MAX;
  u32 ordinal = 0;
  if (static_ordinal_from_raw(params, handles[index], ordinal) &&
      ordinal < params.num_nodes) {
    expected = scalar_reference(
      query_lut,
      params.pq_codes + static_cast<size_t>(ordinal) * params.pq_subquantizers,
      params.pq_subquantizers);
  }
  if (__float_as_uint(expected) != __float_as_uint(distances[index])) {
    atomicAdd(mismatches, 1u);
  }
}

template <u32 PqSubquantizers>
void run_tiled_case(u32 subquantizers, u32 count) {
  const u32 node_count = count;
  std::vector<f32> query_lut = make_query_lut(subquantizers);
  std::vector<u8> codes(static_cast<size_t>(node_count) * subquantizers);
  for (u32 index = 0; index < codes.size(); ++index) {
    codes[index] = static_cast<u8>((index * 73u + 19u) & 0xffu);
  }
  // Handles encode one fake shard whose node records are 16 bytes apart, so
  // candidate i carries raw handle i + 1 (raw 0 is the invalid static
  // encoding).  Sprinkle invalid and out-of-range handles over the wave to
  // pin the FLT_MAX sentinel behavior.
  std::vector<u64> handles(count);
  for (u32 index = 0; index < count; ++index) {
    handles[index] = static_cast<u64>(index) + 1u;
  }
  if (count > 130) {
    handles[7] = gpu_search::kInvalidDeviceHandle;
    handles[128] = static_cast<u64>(node_count) + 5u;  // ordinal >= num_nodes
    handles[129] = handles[130];                       // static duplicate
  }

  DeviceShardRegion shard{};
  shard.ordinal_base = 0;
  shard.node_count = node_count;
  shard.node_base_offset = 16;
  shard.node_stride = 16;

  DeviceBuffer<f32> device_lut(query_lut.size());
  DeviceBuffer<u8> device_codes(codes.size());
  DeviceBuffer<u64> device_handles(handles.size());
  DeviceBuffer<f32> device_distances(count);
  DeviceBuffer<u32> device_mismatches(1);
  DeviceBuffer<DeviceShardRegion> device_shards(1);
  check_cuda(cudaMemcpy(device_lut.get(), query_lut.data(),
                        query_lut.size() * sizeof(f32),
                        cudaMemcpyHostToDevice), "cudaMemcpy LUT H2D");
  check_cuda(cudaMemcpy(device_codes.get(), codes.data(),
                        codes.size() * sizeof(u8), cudaMemcpyHostToDevice),
             "cudaMemcpy codes H2D");
  check_cuda(cudaMemcpy(device_handles.get(), handles.data(),
                        handles.size() * sizeof(u64), cudaMemcpyHostToDevice),
             "cudaMemcpy handles H2D");
  check_cuda(cudaMemcpy(device_shards.get(), &shard, sizeof(shard),
                        cudaMemcpyHostToDevice), "cudaMemcpy shard H2D");
  check_cuda(cudaMemset(device_mismatches.get(), 0, sizeof(u32)),
             "cudaMemset mismatches");

  PersistentKernelParams params{};
  params.shards = device_shards.get();
  params.num_shards = 1;
  params.num_nodes = node_count;
  params.pq_codes = device_codes.get();
  params.pq_subquantizers = subquantizers;
  tiled_scoring_kernel<PqSubquantizers>
    <<<1, 128, gpu_search::kPersistentPqLutTileBytes>>>(
    params, device_lut.get(), device_handles.get(), count,
    device_distances.get());
  check_cuda(cudaGetLastError(), "tiled_scoring_kernel launch");
  compare_tiled_kernel<<<(count + 127u) / 128u, 128>>>(
    params, device_lut.get(), device_handles.get(), count,
    device_distances.get(), device_mismatches.get());
  check_cuda(cudaGetLastError(), "compare_tiled_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "tiled kernels completion");

  u32 mismatches = 0;
  check_cuda(cudaMemcpy(&mismatches, device_mismatches.get(), sizeof(u32),
                        cudaMemcpyDeviceToHost), "cudaMemcpy mismatches D2H");
  if (mismatches != 0) {
    throw std::runtime_error(
      "tiled PQ" + std::to_string(subquantizers) +
      " scoring changed scalar distance results for " +
      std::to_string(mismatches) + " of " + std::to_string(count) +
      " candidates");
  }
}

}  // namespace

int main() {
  try {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count == 0) {
      std::cout << "SKIP: no CUDA device available\n";
      return 0;
    }
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    run_case<gpu_search::kPersistentPq20Subquantizers>(20);
    run_case<gpu_search::kPersistentPq25Subquantizers>(25);
    run_case<gpu_search::kPersistentPq32Subquantizers>(32);
    run_case<gpu_search::kPersistentPq64Subquantizers>(64);
    run_case<gpu_search::kPersistentPq128Subquantizers>(128);
    run_case<gpu_search::kPersistentPq256Subquantizers>(256);
    run_case<gpu_search::kPersistentRuntimePqSubquantizers>(17);
    run_case<gpu_search::kPersistentRuntimePqSubquantizers>(64);
    // Tiled shared-LUT scoring: full and partial waves, an odd runtime width
    // (byte tail path), and every production packed specialization.
    run_tiled_case<gpu_search::kPersistentPq64Subquantizers>(64, 2048);
    run_tiled_case<gpu_search::kPersistentPq64Subquantizers>(64, 1379);
    run_tiled_case<gpu_search::kPersistentPq128Subquantizers>(128, 2048);
    run_tiled_case<gpu_search::kPersistentPq256Subquantizers>(256, 2048);
    run_tiled_case<gpu_search::kPersistentPq256Subquantizers>(256, 1379);
    run_tiled_case<gpu_search::kPersistentPq32Subquantizers>(32, 2048);
    run_tiled_case<gpu_search::kPersistentPq20Subquantizers>(20, 1379);
    run_tiled_case<gpu_search::kPersistentRuntimePqSubquantizers>(17, 2048);
    run_tiled_case<gpu_search::kPersistentRuntimePqSubquantizers>(25, 1379);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
