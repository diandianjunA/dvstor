#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_search/host_distance_kernel.hh"

namespace {

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <class T>
T* allocate(std::size_t count) {
  T* result = nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&result), count * sizeof(T)),
        "cudaMalloc");
  return result;
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cout << "SKIP: CUDA device unavailable\n";
    return 0;
  }
  check(cudaSetDevice(0), "cudaSetDevice");
  cudaStream_t stream{};
  check(cudaStreamCreate(&stream), "cudaStreamCreate");

  constexpr std::uint32_t kCodeBytes = 2;
  const std::vector<std::uint8_t> resident{1, 2, 3, 4};
  const std::vector<std::uint32_t> ordinals{
    1, std::numeric_limits<std::uint32_t>::max()};
  const std::vector<std::uint8_t> dynamic{0, 0, 5, 6};
  std::vector<float> lut(kCodeBytes * 256u, 0.0f);
  for (std::uint32_t code = 0; code < 256; ++code) {
    lut[code] = static_cast<float>(code);
    lut[256u + code] = 10.0f * static_cast<float>(code);
  }
  auto* d_resident = allocate<std::uint8_t>(resident.size());
  auto* d_ordinals = allocate<std::uint32_t>(ordinals.size());
  auto* d_dynamic = allocate<std::uint8_t>(dynamic.size());
  auto* d_lut = allocate<float>(lut.size());
  auto* d_distances = allocate<float>(2);
  check(cudaMemcpy(d_resident, resident.data(), resident.size(),
                   cudaMemcpyHostToDevice), "copy resident");
  check(cudaMemcpy(d_ordinals, ordinals.data(),
                   ordinals.size() * sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice), "copy ordinals");
  check(cudaMemcpy(d_dynamic, dynamic.data(), dynamic.size(),
                   cudaMemcpyHostToDevice), "copy dynamic");
  check(cudaMemcpy(d_lut, lut.data(), lut.size() * sizeof(float),
                   cudaMemcpyHostToDevice), "copy LUT");
  gpu_search::host_distance::launch_pq(
    stream, d_resident, d_ordinals, d_dynamic, 2, kCodeBytes, d_lut,
    d_distances);
  check(cudaGetLastError(), "launch PQ");
  std::vector<float> distances(2);
  check(cudaMemcpyAsync(distances.data(), d_distances,
                        distances.size() * sizeof(float),
                        cudaMemcpyDeviceToHost, stream), "copy PQ result");
  check(cudaStreamSynchronize(stream), "sync PQ");
  assert(std::abs(distances[0] - 43.0f) < 1e-5f);
  assert(std::abs(distances[1] - 65.0f) < 1e-5f);

  const std::vector<float> query{1, 2, 3, 4};
  const std::vector<std::uint8_t> records{
    1, 2, 3, 4, 0, 0, 0, 0,
    2, 4, 6, 8, 0, 0, 0, 0};
  auto* d_query = allocate<float>(query.size());
  auto* d_records = allocate<std::uint8_t>(records.size());
  check(cudaMemcpy(d_query, query.data(), query.size() * sizeof(float),
                   cudaMemcpyHostToDevice), "copy exact query");
  check(cudaMemcpy(d_records, records.data(), records.size(),
                   cudaMemcpyHostToDevice), "copy exact records");
  gpu_search::host_distance::launch_exact(
    stream, d_query, d_records, 2, 4,
    static_cast<VectorDType>(1), 8, 0, d_distances);
  check(cudaGetLastError(), "launch exact");
  check(cudaMemcpyAsync(distances.data(), d_distances,
                        distances.size() * sizeof(float),
                        cudaMemcpyDeviceToHost, stream), "copy exact result");
  check(cudaStreamSynchronize(stream), "sync exact");
  assert(std::abs(distances[0]) < 1e-5f);
  assert(std::abs(distances[1] - 30.0f) < 1e-5f);

  cudaFree(d_records);
  cudaFree(d_query);
  cudaFree(d_distances);
  cudaFree(d_lut);
  cudaFree(d_dynamic);
  cudaFree(d_ordinals);
  cudaFree(d_resident);
  cudaStreamDestroy(stream);
  return 0;
}
