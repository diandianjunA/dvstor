#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_search/persistent_kernel/candidate_scoring.cuh"

namespace {

using gpu_search::BeamMergePolicy;
using gpu_search::f32;
using gpu_search::u8;
using gpu_search::u32;
using gpu_search::u64;
using namespace gpu_search::persistent_kernel_detail;

constexpr u32 kWarmupIterations = 8;
constexpr u32 kMeasuredIterations = 64;

struct MicrobenchResult {
  u64 total_cycles{};
  u64 prepare_cycles{};
  u64 sort_cycles{};
  u64 materialize_cycles{};
};

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                          count * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const { return data_; }

 private:
  T* data_{};
};

template <class T>
void upload(DeviceBuffer<T>& destination, const std::vector<T>& source) {
  check_cuda(cudaMemcpy(destination.get(), source.data(),
                        source.size() * sizeof(T), cudaMemcpyHostToDevice),
             "cudaMemcpy H2D");
}

template <class T>
T download_one(const DeviceBuffer<T>& source) {
  T result{};
  check_cuda(cudaMemcpy(&result, source.get(), sizeof(T),
                        cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  return result;
}

__global__ void stable_run_microbench_kernel(
    const u64* candidate_handles, const f32* candidate_distances,
    u32 candidate_count, const u64* original_beam_handles,
    const f32* original_beam_distances, const u8* original_beam_expanded,
    u32 beam_capacity, u64* beam_handles, u32* beam_ids,
    f32* beam_distances, u8* beam_expanded, u64* scratch_handles,
    u8* scratch_flags, f32* scratch_distances, BeamMergePolicy policy,
    bool fused_materialize, MicrobenchResult* result) {
  __shared__ CandidateWorkspace workspace;
  __shared__ StableMergePreparedState stable_state;
  __shared__ u32 beam_count;
  __shared__ u64 iteration_started;
  __shared__ BeamMergeCycleBreakdown phases;

  u64 total_cycles = 0;
  u64 prepare_cycles = 0;
  u64 sort_cycles = 0;
  u64 materialize_cycles = 0;
  for (u32 iteration = 0;
       iteration < kWarmupIterations + kMeasuredIterations; ++iteration) {
    for (u32 index = threadIdx.x; index < beam_capacity;
         index += blockDim.x) {
      beam_handles[index] = original_beam_handles[index];
      beam_ids[index] = UINT32_MAX;
      beam_distances[index] = original_beam_distances[index];
      beam_expanded[index] = original_beam_expanded[index];
    }
    if (threadIdx.x == 0) {
      beam_count = beam_capacity;
      phases = {};
    }
    __syncthreads();
    if (threadIdx.x == 0) iteration_started = clock64();
    __syncthreads();

    if (policy == BeamMergePolicy::stable_run && fused_materialize) {
      prepare_approximate_stable_runs(
        const_cast<u64*>(candidate_handles),
        const_cast<f32*>(candidate_distances), candidate_count,
        beam_handles, beam_ids, beam_distances, beam_expanded,
        beam_count, beam_capacity,
        scratch_handles, scratch_flags, scratch_distances,
        workspace, stable_state, &phases, false);
      finish_approximate_stable_runs(
        beam_handles, beam_ids, beam_distances, beam_expanded,
        beam_count, beam_capacity,
        scratch_handles, scratch_flags, scratch_distances,
        workspace, stable_state, &phases, true);
    } else {
      merge_approximate_into_beam(
        const_cast<u64*>(candidate_handles),
        const_cast<f32*>(candidate_distances), candidate_count,
        beam_handles, beam_ids, beam_distances, beam_expanded,
        beam_count, beam_capacity,
        scratch_handles, scratch_flags, scratch_distances,
        workspace, policy,
        policy == BeamMergePolicy::stable_run ? &phases : nullptr);
    }

    if (threadIdx.x == 0 && iteration >= kWarmupIterations) {
      total_cycles += clock64() - iteration_started;
      prepare_cycles += phases.prepare;
      sort_cycles += phases.sort;
      materialize_cycles += phases.materialize;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *result = {
      total_cycles, prepare_cycles, sort_cycles, materialize_cycles};
  }
}

MicrobenchResult run_one(
    u32 threads, u32 beam_capacity, u32 candidate_count,
    BeamMergePolicy policy, bool fused_materialize) {
  if (beam_capacity + candidate_count >
      gpu_search::kPersistentMaxMergeCandidates) {
    throw std::invalid_argument("microbench merge input exceeds capacity");
  }
  std::vector<u64> candidate_handles(candidate_count);
  std::vector<f32> candidate_distances(candidate_count);
  for (u32 index = 0; index < candidate_count; ++index) {
    candidate_handles[index] = 0x200000000ULL + index;
    // Deliberately unsorted with repeated keys, exercising stable candidate
    // ordering without invalid data dominating the timing.
    candidate_distances[index] =
      static_cast<f32>((index * 37u + 11u) % 1024u) * 0.25f;
  }
  std::vector<u64> original_beam_handles(beam_capacity);
  std::vector<f32> original_beam_distances(beam_capacity);
  std::vector<u8> original_beam_expanded(beam_capacity);
  for (u32 index = 0; index < beam_capacity; ++index) {
    original_beam_handles[index] = 0x100000000ULL + index;
    original_beam_distances[index] = static_cast<f32>(index) + 0.125f;
    original_beam_expanded[index] = static_cast<u8>((index % 5u) == 0);
  }

  DeviceBuffer<u64> d_candidate_handles(candidate_count);
  DeviceBuffer<f32> d_candidate_distances(candidate_count);
  DeviceBuffer<u64> d_original_beam_handles(beam_capacity);
  DeviceBuffer<f32> d_original_beam_distances(beam_capacity);
  DeviceBuffer<u8> d_original_beam_expanded(beam_capacity);
  DeviceBuffer<u64> d_beam_handles(beam_capacity);
  DeviceBuffer<u32> d_beam_ids(beam_capacity);
  DeviceBuffer<f32> d_beam_distances(beam_capacity);
  DeviceBuffer<u8> d_beam_expanded(beam_capacity);
  DeviceBuffer<u64> d_scratch_handles(beam_capacity * 4);
  DeviceBuffer<u8> d_scratch_flags(beam_capacity * 4);
  DeviceBuffer<f32> d_scratch_distances(beam_capacity * 4);
  DeviceBuffer<MicrobenchResult> d_result(1);
  upload(d_candidate_handles, candidate_handles);
  upload(d_candidate_distances, candidate_distances);
  upload(d_original_beam_handles, original_beam_handles);
  upload(d_original_beam_distances, original_beam_distances);
  upload(d_original_beam_expanded, original_beam_expanded);

  stable_run_microbench_kernel<<<1, threads>>>(
    d_candidate_handles.get(), d_candidate_distances.get(), candidate_count,
    d_original_beam_handles.get(), d_original_beam_distances.get(),
    d_original_beam_expanded.get(), beam_capacity,
    d_beam_handles.get(), d_beam_ids.get(), d_beam_distances.get(),
    d_beam_expanded.get(), d_scratch_handles.get(), d_scratch_flags.get(),
    d_scratch_distances.get(), policy, fused_materialize, d_result.get());
  check_cuda(cudaGetLastError(), "stable_run_microbench_kernel launch");
  check_cuda(cudaDeviceSynchronize(),
             "stable_run_microbench_kernel completion");
  const MicrobenchResult result = download_one(d_result);
  if (result.total_cycles == 0 ||
      (policy == BeamMergePolicy::stable_run &&
       (result.sort_cycles == 0 || result.materialize_cycles == 0))) {
    throw std::runtime_error("merge microbench did not record device cycles");
  }
  return result;
}

void print_result(
    u32 threads, u32 beam_capacity, u32 candidate_count,
    BeamMergePolicy policy, bool fused_materialize,
    const MicrobenchResult& result,
    double cycles_per_microsecond) {
  const char* name =
    policy == BeamMergePolicy::legacy ? "legacy" : "stable-run";
  const double divisor = static_cast<double>(kMeasuredIterations);
  const double total = static_cast<double>(result.total_cycles) / divisor;
  const double prepare = static_cast<double>(result.prepare_cycles) / divisor;
  const double sort = static_cast<double>(result.sort_cycles) / divisor;
  const double materialize =
    static_cast<double>(result.materialize_cycles) / divisor;
  std::cout << "merge_microbench"
            << " policy=" << name
            << " materializer="
            << (fused_materialize ? "fused-tree" : "default")
            << " threads=" << threads
            << " beam=" << beam_capacity
            << " candidates=" << candidate_count
            << " total_cycles=" << std::fixed << std::setprecision(1) << total
            << " total_us~=" << std::setprecision(3)
            << total / cycles_per_microsecond;
  if (policy == BeamMergePolicy::stable_run) {
    std::cout << " prepare_cycles=" << std::setprecision(1) << prepare
              << " sort_cycles=" << sort
              << " materialize_cycles=" << materialize;
  }
  std::cout << '\n';
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
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, 0),
               "cudaGetDeviceProperties");
    const double cycles_per_microsecond =
      static_cast<double>(properties.clockRate) / 1000.0;

    for (const u32 threads : {128u, 256u}) {
      for (const u32 beam_capacity : {128u, 256u}) {
        for (const u32 candidate_count : {512u, 1536u}) {
          if (beam_capacity + candidate_count >
              gpu_search::kPersistentMaxMergeCandidates) {
            continue;
          }
          for (const BeamMergePolicy policy :
               {BeamMergePolicy::legacy, BeamMergePolicy::stable_run}) {
            const u32 materializer_count =
              policy == BeamMergePolicy::stable_run &&
                threads == kApproximateSortThreadsCompact &&
                beam_capacity <= gpu_search::kPersistentMaxBeam
                ? 2u : 1u;
            for (u32 materializer = 0; materializer < materializer_count;
                 ++materializer) {
              const bool fused_materialize = materializer != 0;
              const MicrobenchResult result = run_one(
                threads, beam_capacity, candidate_count, policy,
                fused_materialize);
              print_result(
                threads, beam_capacity, candidate_count, policy,
                fused_materialize, result, cycles_per_microsecond);
            }
          }
        }
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
