#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_search/persistent_kernel/candidate_scoring.cuh"

namespace {

using gpu_search::BeamMergePolicy;
using gpu_search::f32;
using gpu_search::u8;
using gpu_search::u16;
using gpu_search::u32;
using gpu_search::u64;
using namespace gpu_search::persistent_kernel_detail;

struct PreviewResult {
  u32 count{};
  u32 mismatch{};
  u64 serial_cycles{};
  u64 parallel_cycles{};
};

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                          count * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }
  T* get() const { return data_; }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

 private:
  T* data_{};
};

template <typename T>
void upload(DeviceBuffer<T>& destination, const std::vector<T>& source) {
  check_cuda(cudaMemcpy(destination.get(), source.data(),
                        source.size() * sizeof(T), cudaMemcpyHostToDevice),
             "cudaMemcpy H2D");
}

template <typename T>
std::vector<T> download(const DeviceBuffer<T>& source, size_t count) {
  std::vector<T> result(count);
  check_cuda(cudaMemcpy(result.data(), source.get(), count * sizeof(T),
                        cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  return result;
}

__device__ void serial_preview_reference(
    const u64* beam_handles, const f32* beam_distances,
    const u8* beam_expanded, u32 beam_count, u32 beam_capacity,
    const u64* scratch_handles, const u8* scratch_flags,
    const f32* scratch_distances, u32 candidate_run_count,
    u32 issue_capacity, u64* output_handles, u16* output_ranks,
    u32& output_count) {
  if (threadIdx.x == 0) {
    u32 heads[5]{0, 0, 0, 0, 0};
    candidate_run_count = min(candidate_run_count, 4u);
    const u32 run_count = candidate_run_count + 1u;
    output_count = 0;
    for (u32 rank = 0;
         rank < beam_capacity && output_count < issue_capacity; ++rank) {
      u32 selected_run = UINT32_MAX;
      f32 selected_distance = FLT_MAX;
      for (u32 run = 0; run < run_count; ++run) {
        const u32 head = heads[run];
        const u32 count = run == 0 ? beam_count : beam_capacity;
        if (head >= count) continue;
        const u64 handle = run == 0
          ? beam_handles[head]
          : scratch_handles[(run - 1u) * beam_capacity + head];
        const f32 distance = run == 0
          ? beam_distances[head]
          : scratch_distances[(run - 1u) * beam_capacity + head];
        if (!stable_run_item_valid(handle, distance)) continue;
        if (selected_run == UINT32_MAX ||
            stable_run_head_precedes(
              distance, run, selected_distance, selected_run)) {
          selected_run = run;
          selected_distance = distance;
        }
      }
      if (selected_run == UINT32_MAX) break;
      const u32 head = heads[selected_run]++;
      const u64 handle = selected_run == 0
        ? beam_handles[head]
        : scratch_handles[
            (selected_run - 1u) * beam_capacity + head];
      const bool expanded = selected_run == 0
        ? beam_expanded[head] != 0
        : scratch_flags[
            (selected_run - 1u) * beam_capacity + head] != 0;
      if (expanded) continue;
      output_handles[output_count] = handle;
      output_ranks[output_count] = static_cast<u16>(rank);
      ++output_count;
    }
  }
  __syncthreads();
}

__global__ void preview_kernel(
    const u64* candidate_handles, const f32* candidate_distances,
    u32 candidate_count, u64* beam_handles, u32* beam_ids,
    f32* beam_distances, u8* beam_expanded, u32 beam_capacity,
    u64* scratch_handles, u8* scratch_flags, f32* scratch_distances,
    u32 issue_capacity, u64* preview_handles, u16* preview_ranks,
    u64* reference_handles, u16* reference_ranks,
    PreviewResult* result) {
  __shared__ CandidateWorkspace workspace;
  __shared__ StableMergePreparedState state;
  __shared__ u32 beam_count;
  __shared__ u32 preview_count;
  __shared__ u32 reference_count;
  __shared__ BeamMergeCycleBreakdown phases;
  __shared__ u64 phase_start;
  __shared__ u64 serial_cycles;
  __shared__ u64 parallel_cycles;

  for (u32 index = threadIdx.x; index < beam_capacity; index += blockDim.x) {
    beam_handles[index] = 0x100000000ULL + index;
    beam_ids[index] = index;
    // Stable-Run's authoritative old Beam is already sorted. Repeated
    // distances exercise its old-run-first tie contract.
    beam_distances[index] = static_cast<f32>(index / 4u) * 0.25f;
    beam_expanded[index] = static_cast<u8>((index % 7u) == 0);
  }
  if (threadIdx.x == 0) {
    beam_count = beam_capacity;
    preview_count = 0;
    reference_count = 0;
    phases = {};
    serial_cycles = 0;
    parallel_cycles = 0;
    *result = {};
  }
  __syncthreads();

  prepare_approximate_stable_runs(
    const_cast<u64*>(candidate_handles),
    const_cast<f32*>(candidate_distances), candidate_count,
    beam_handles, beam_ids, beam_distances, beam_expanded,
    beam_count, beam_capacity, scratch_handles, scratch_flags,
    scratch_distances, workspace, state, &phases, false);
  if (threadIdx.x == 0) phase_start = clock64();
  __syncthreads();
  serial_preview_reference(
    beam_handles, beam_distances, beam_expanded, beam_count, beam_capacity,
    scratch_handles, scratch_flags, scratch_distances,
    state.candidate_run_count, issue_capacity,
    reference_handles, reference_ranks, reference_count);
  if (threadIdx.x == 0) {
    serial_cycles = clock64() - phase_start;
  }
  __syncthreads();
  if (threadIdx.x == 0) phase_start = clock64();
  __syncthreads();
  preview_tree_stable_unexpanded_frontier(
    beam_handles, beam_distances, beam_expanded,
    beam_count, beam_capacity,
    scratch_handles, scratch_distances,
    state.candidate_run_count, issue_capacity, workspace.arrays,
    preview_handles, preview_ranks, preview_count);
  if (threadIdx.x == 0) parallel_cycles = clock64() - phase_start;
  __syncthreads();
  for (u32 output = threadIdx.x; output < reference_count;
       output += blockDim.x) {
    if (output >= preview_count ||
        preview_handles[output] != reference_handles[output] ||
        preview_ranks[output] != reference_ranks[output]) {
      atomicExch(&result->mismatch, 1u);
    }
  }
  __syncthreads();
  finish_approximate_stable_runs(
    beam_handles, beam_ids, beam_distances, beam_expanded,
    beam_count, beam_capacity, scratch_handles, scratch_flags,
    scratch_distances, workspace, state, &phases, true);

  if (threadIdx.x == 0) {
    u32 expected = 0;
    u32 mismatch =
      result->mismatch != 0 || reference_count != preview_count
        ? 1u : 0u;
    for (u32 rank = 0; rank < beam_count; ++rank) {
      if (beam_expanded[rank] != 0) continue;
      if (expected == issue_capacity) break;
      if (expected >= preview_count ||
          preview_handles[expected] != beam_handles[rank] ||
          preview_ranks[expected] != rank) {
        mismatch = 1;
        break;
      }
      ++expected;
    }
    if (expected != preview_count) mismatch = 1;
    *result = PreviewResult{
      preview_count, mismatch, serial_cycles, parallel_cycles};
  }
}

void run_case(u32 threads, u32 capacity, u32 issue_capacity) {
  const u32 candidate_count = capacity == 128 ? 1333u : 777u;
  std::vector<u64> candidates(candidate_count);
  std::vector<f32> distances(candidate_count);
  for (u32 index = 0; index < candidate_count; ++index) {
    candidates[index] = 0x200000000ULL + (index * 13u) % 4096u;
    distances[index] =
      static_cast<f32>((index * 37u + 11u) % 233u) * 0.125f;
  }

  DeviceBuffer<u64> d_candidates(candidate_count);
  DeviceBuffer<f32> d_distances(candidate_count);
  DeviceBuffer<u64> d_beam_handles(capacity);
  DeviceBuffer<u32> d_beam_ids(capacity);
  DeviceBuffer<f32> d_beam_distances(capacity);
  DeviceBuffer<u8> d_beam_expanded(capacity);
  DeviceBuffer<u64> d_scratch_handles(capacity * 4);
  DeviceBuffer<u8> d_scratch_flags(capacity * 4);
  DeviceBuffer<f32> d_scratch_distances(capacity * 4);
  DeviceBuffer<u64> d_preview_handles(capacity);
  DeviceBuffer<u16> d_preview_ranks(capacity);
  DeviceBuffer<u64> d_reference_handles(capacity);
  DeviceBuffer<u16> d_reference_ranks(capacity);
  DeviceBuffer<PreviewResult> d_result(1);
  upload(d_candidates, candidates);
  upload(d_distances, distances);

  preview_kernel<<<1, threads>>>(
    d_candidates.get(), d_distances.get(), candidate_count,
    d_beam_handles.get(), d_beam_ids.get(), d_beam_distances.get(),
    d_beam_expanded.get(), capacity, d_scratch_handles.get(),
    d_scratch_flags.get(), d_scratch_distances.get(),
    issue_capacity, d_preview_handles.get(), d_preview_ranks.get(),
    d_reference_handles.get(), d_reference_ranks.get(), d_result.get());
  check_cuda(cudaGetLastError(), "preview_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "preview_kernel synchronize");
  const auto result = download(d_result, 1);
  if (result[0].mismatch != 0) {
    throw std::runtime_error(
      "post-score Stable-Run preview diverged from materialized frontier");
  }
  std::cout << "threads=" << threads << ",capacity=" << capacity
            << ",issue_capacity=" << issue_capacity
            << ",preview_count=" << result[0].count
            << ",serial_cycles=" << result[0].serial_cycles
            << ",parallel_cycles=" << result[0].parallel_cycles << '\n';
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: no CUDA device available\n";
    return 0;
  }
  try {
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    for (const u32 issue_capacity : {1u, 8u, 16u, 32u}) {
      run_case(128, 64, issue_capacity);
      run_case(128, 128, issue_capacity);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
