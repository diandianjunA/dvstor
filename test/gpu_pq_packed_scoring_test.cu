#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_search/persistent_kernel/candidate_scoring.cuh"

namespace {

using gpu_search::PersistentKernelParams;
using gpu_search::f32;
using gpu_search::u8;
using gpu_search::u32;
using gpu_search::persistent_kernel_detail::approximate_entry;

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

template <u32 PqSubquantizers>
void run_case(u32 subquantizers) {
  constexpr u32 kCases = 4096;
  constexpr u32 kMaximumSubquantizers = 32;
  std::vector<f32> query_lut(kMaximumSubquantizers * 256u);
  for (u32 subquantizer = 0; subquantizer < kMaximumSubquantizers;
       ++subquantizer) {
    for (u32 centroid = 0; centroid < 256; ++centroid) {
      // Binary fractions keep the equality assertion sensitive to addition
      // order without introducing host/device decimal conversion noise.
      query_lut[static_cast<size_t>(subquantizer) * 256u + centroid] =
        static_cast<f32>((subquantizer * 257u + centroid) & 0xffffu) /
        1024.0f;
    }
  }
  std::vector<u8> codes(
    static_cast<size_t>(kCases) * subquantizers +
    kMaximumSubquantizers);
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
    run_case<gpu_search::kPersistentRuntimePqSubquantizers>(17);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
