#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_search/persistent_kernel/candidate_scoring.cuh"

namespace {

using gpu_search::f32;
using gpu_search::u8;
using gpu_search::u32;
using gpu_search::u64;
using gpu_search::persistent_kernel_detail::CandidateWorkspace;
using gpu_search::persistent_kernel_detail::merge_approximate_compact;

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

template <class T>
void copy_to_device(DeviceBuffer<T>& destination,
                    const std::vector<T>& source) {
  if (destination.size() != source.size()) {
    throw std::invalid_argument("host/device test buffer size mismatch");
  }
  check_cuda(cudaMemcpy(destination.get(), source.data(),
                        source.size() * sizeof(T), cudaMemcpyHostToDevice),
             "cudaMemcpy(host to device)");
}

template <class T>
std::vector<T> copy_from_device(const DeviceBuffer<T>& source) {
  std::vector<T> result(source.size());
  check_cuda(cudaMemcpy(result.data(), source.get(),
                        result.size() * sizeof(T), cudaMemcpyDeviceToHost),
             "cudaMemcpy(device to host)");
  return result;
}

__global__ void compact_merge_kernel(
    u64* candidate_handles, f32* candidate_distances, u32 candidate_count,
    u64* beam_handles, u32* beam_ids, f32* beam_distances,
    u8* beam_expanded, u32 beam_capacity,
    u64* scratch_handles, u32* scratch_expanded, f32* scratch_distances,
    u32* result_count) {
  __shared__ CandidateWorkspace workspace;
  __shared__ u32 beam_count;
  if (threadIdx.x == 0) beam_count = beam_capacity;
  __syncthreads();
  merge_approximate_compact(
    candidate_handles, candidate_distances,
    beam_handles, beam_ids, beam_distances, beam_expanded,
    beam_count, beam_capacity, beam_capacity,
    beam_capacity + candidate_count,
    scratch_handles, scratch_expanded, scratch_distances, workspace);
  if (threadIdx.x == 0) *result_count = beam_count;
}

void run_case(u32 beam_capacity) {
  constexpr u32 kMergeItems = gpu_search::kPersistentMaxMergeCandidates;
  constexpr u32 kPassItems =
    gpu_search::persistent_kernel_detail::kApproximateSortThreadsCompact *
    gpu_search::persistent_kernel_detail::kApproximateSortItemsCompactPass;
  if (beam_capacity != 128 && beam_capacity != 256) {
    throw std::invalid_argument("unexpected compact beam test capacity");
  }
  const u32 candidate_count = kMergeItems - beam_capacity;
  const u32 scratch_count = beam_capacity * 2;

  std::vector<u64> candidate_handles(candidate_count);
  std::vector<f32> candidate_distances(candidate_count);
  for (u32 index = 0; index < candidate_count; ++index) {
    candidate_handles[index] = 0x200000000ULL + index;
    const u32 combined_index = beam_capacity + index;
    candidate_distances[index] = combined_index >= kPassItems
      ? static_cast<f32>(combined_index - kPassItems)
      : static_cast<f32>(100000 + combined_index);
  }

  std::vector<u64> beam_handles(beam_capacity);
  std::vector<u32> beam_ids(beam_capacity, 7);
  std::vector<f32> beam_distances(beam_capacity);
  std::vector<u8> beam_expanded(beam_capacity, 0);
  for (u32 index = 0; index < beam_capacity; ++index) {
    beam_handles[index] = 0x100000000ULL + index;
    beam_distances[index] = static_cast<f32>(200000 + index);
  }
  // Keep one expanded entry globally best.  This also verifies that the
  // two-pass top-k merge preserves expansion state while selecting the best
  // candidates from the second 1024-item pass.
  beam_distances[0] = -1.0f;
  beam_expanded[0] = 1;

  DeviceBuffer<u64> d_candidate_handles(candidate_count);
  DeviceBuffer<f32> d_candidate_distances(candidate_count);
  DeviceBuffer<u64> d_beam_handles(beam_capacity);
  DeviceBuffer<u32> d_beam_ids(beam_capacity);
  DeviceBuffer<f32> d_beam_distances(beam_capacity);
  DeviceBuffer<u8> d_beam_expanded(beam_capacity);
  DeviceBuffer<u64> d_scratch_handles(scratch_count);
  DeviceBuffer<u32> d_scratch_expanded(scratch_count);
  DeviceBuffer<f32> d_scratch_distances(scratch_count);
  DeviceBuffer<u32> d_result_count(1);
  copy_to_device(d_candidate_handles, candidate_handles);
  copy_to_device(d_candidate_distances, candidate_distances);
  copy_to_device(d_beam_handles, beam_handles);
  copy_to_device(d_beam_ids, beam_ids);
  copy_to_device(d_beam_distances, beam_distances);
  copy_to_device(d_beam_expanded, beam_expanded);

  compact_merge_kernel<<<1, 128>>>(
    d_candidate_handles.get(), d_candidate_distances.get(), candidate_count,
    d_beam_handles.get(), d_beam_ids.get(), d_beam_distances.get(),
    d_beam_expanded.get(), beam_capacity,
    d_scratch_handles.get(), d_scratch_expanded.get(),
    d_scratch_distances.get(), d_result_count.get());
  check_cuda(cudaGetLastError(), "compact_merge_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "compact_merge_kernel completion");

  const auto output_handles = copy_from_device(d_beam_handles);
  const auto output_ids = copy_from_device(d_beam_ids);
  const auto output_distances = copy_from_device(d_beam_distances);
  const auto output_expanded = copy_from_device(d_beam_expanded);
  const auto output_count = copy_from_device(d_result_count);
  if (output_count[0] != beam_capacity ||
      output_handles[0] != beam_handles[0] ||
      output_distances[0] != -1.0f || output_expanded[0] != 1) {
    throw std::runtime_error(
      "compact merge lost the best existing beam entry");
  }
  const u32 second_pass_candidate = kPassItems - beam_capacity;
  for (u32 output = 1; output < beam_capacity; ++output) {
    const u32 candidate = second_pass_candidate + output - 1;
    if (output_handles[output] != candidate_handles[candidate] ||
        output_distances[output] != static_cast<f32>(output - 1) ||
        output_expanded[output] != 0 ||
        output_ids[output] != std::numeric_limits<u32>::max()) {
      throw std::runtime_error(
        "compact merge did not retain the global top-k across both passes");
    }
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
    run_case(128);
    run_case(256);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
