#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
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
using namespace gpu_search::persistent_kernel_detail;

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
  T* get() const { return data_; }
  size_t size() const { return count_; }
 private:
  T* data_{};
  size_t count_{};
};

template <class T>
void upload(DeviceBuffer<T>& destination, const std::vector<T>& source) {
  check_cuda(cudaMemcpy(destination.get(), source.data(),
                        source.size() * sizeof(T), cudaMemcpyHostToDevice),
             "cudaMemcpy H2D");
}

template <class T>
std::vector<T> download(const DeviceBuffer<T>& source) {
  std::vector<T> result(source.size());
  check_cuda(cudaMemcpy(result.data(), source.get(),
                        result.size() * sizeof(T), cudaMemcpyDeviceToHost),
             "cudaMemcpy D2H");
  return result;
}

__global__ void feedback_merge_kernel(
    u64* candidate_handles, f32* candidate_distances, u32 candidate_count,
    u64* beam_handles, u32* beam_ids, f32* beam_distances,
    u8* beam_expanded, u32 initial_beam_count, u32 beam_capacity,
    u64* scratch_handles, u32* scratch_ids, f32* scratch_distances,
    u8* scratch_expanded, u64* compact_handles,
    u32* compact_flags, f32* compact_distances,
    FeedbackHorizonResult* feedback, u32 feedback_cap, u32* output_count) {
  __shared__ CandidateWorkspace workspace;
  __shared__ u32 beam_count;
  if (threadIdx.x == 0) beam_count = initial_beam_count;
  __syncthreads();
  merge_approximate_into_beam(
    candidate_handles, candidate_distances, candidate_count,
    beam_handles, beam_ids, beam_distances, beam_expanded,
    beam_count, beam_capacity, scratch_handles, scratch_ids,
    scratch_distances, scratch_expanded, compact_handles,
    compact_flags, compact_distances, workspace, feedback_cap, feedback);
  if (threadIdx.x == 0) *output_count = beam_count;
}

struct HostItem {
  u64 handle;
  f32 distance;
  u32 input;
  bool matched_old;
  u8 expanded;
};

struct HostReference {
  std::vector<HostItem> beam;
  FeedbackHorizonResult feedback;
};

HostReference reference_merge(
    const std::vector<u64>& old_handles,
    const std::vector<f32>& old_distances,
    const std::vector<u8>& old_expanded,
    const std::vector<u64>& candidate_handles,
    const std::vector<f32>& candidate_distances,
    u32 beam_capacity, u32 feedback_cap) {
  std::vector<HostItem> items;
  for (u32 index = 0; index < old_handles.size(); ++index) {
    items.push_back({
      old_handles[index], old_distances[index], index, true,
      old_expanded[index]});
  }
  for (u32 index = 0; index < candidate_handles.size(); ++index) {
    u64 handle = candidate_handles[index];
    f32 distance = candidate_distances[index];
    if (handle == gpu_search::kInvalidDeviceHandle ||
        !std::isfinite(distance)) {
      handle = gpu_search::kInvalidDeviceHandle;
      distance = FLT_MAX;
    }
    bool matched_old = false;
    u8 expanded = 0;
    for (u32 prior = 0; prior < old_handles.size(); ++prior) {
      if (old_handles[prior] == handle) {
        matched_old = true;
        expanded = old_expanded[prior];
        break;
      }
    }
    items.push_back({
      handle, distance,
      static_cast<u32>(old_handles.size() + index),
      matched_old, expanded});
  }
  std::stable_sort(items.begin(), items.end(),
    [](const HostItem& lhs, const HostItem& rhs) {
      return lhs.distance < rhs.distance;
    });
  HostReference result;
  u32 earliest_new = UINT32_MAX;
  u32 new_count = 0;
  for (const HostItem& item : items) {
    if (result.beam.size() == beam_capacity) break;
    if (item.handle == gpu_search::kInvalidDeviceHandle ||
        !std::isfinite(item.distance) || item.distance == FLT_MAX) {
      break;
    }
    const u32 output = static_cast<u32>(result.beam.size());
    result.beam.push_back(item);
    if (!item.matched_old) {
      earliest_new = std::min(earliest_new, output);
      ++new_count;
    }
  }
  u32 unexpanded_before = 0;
  u32 unexpanded_total = 0;
  for (u32 index = 0; index < result.beam.size(); ++index) {
    if (result.beam[index].expanded == 0) {
      ++unexpanded_total;
      if (index < earliest_new) ++unexpanded_before;
    }
  }
  result.feedback = {
    .horizon = earliest_new < result.beam.size()
      ? std::min(unexpanded_before + 1u, unexpanded_total)
      : std::min(unexpanded_total, feedback_cap),
    .earliest_new_output = earliest_new,
    .old_unexpanded_before_new = unexpanded_before,
    .new_candidates_in_beam = new_count,
  };
  return result;
}

void run_case(
    u32 threads, u32 beam_capacity,
    std::vector<u64> old_handles, std::vector<f32> old_distances,
    std::vector<u8> old_expanded,
    std::vector<u64> candidate_handles,
    std::vector<f32> candidate_distances) {
  constexpr u32 feedback_cap = 16;
  const u32 old_count = static_cast<u32>(old_handles.size());
  const HostReference expected = reference_merge(
    old_handles, old_distances, old_expanded,
    candidate_handles, candidate_distances, beam_capacity, feedback_cap);
  std::vector<u32> old_ids(beam_capacity, 7);
  old_handles.resize(beam_capacity, gpu_search::kInvalidDeviceHandle);
  old_distances.resize(beam_capacity, FLT_MAX);
  old_expanded.resize(beam_capacity, 0);

  DeviceBuffer<u64> d_candidates(candidate_handles.size());
  DeviceBuffer<f32> d_candidate_distances(candidate_distances.size());
  DeviceBuffer<u64> d_beam(beam_capacity);
  DeviceBuffer<u32> d_ids(beam_capacity);
  DeviceBuffer<f32> d_distances(beam_capacity);
  DeviceBuffer<u8> d_expanded(beam_capacity);
  DeviceBuffer<u64> d_scratch_handles(beam_capacity * 2);
  DeviceBuffer<u32> d_scratch_ids(beam_capacity * 2);
  DeviceBuffer<f32> d_scratch_distances(beam_capacity * 2);
  DeviceBuffer<u8> d_scratch_expanded(beam_capacity * 2);
  DeviceBuffer<u64> d_compact_handles(beam_capacity * 2);
  DeviceBuffer<u32> d_compact_flags(beam_capacity * 2);
  DeviceBuffer<f32> d_compact_distances(beam_capacity * 2);
  DeviceBuffer<FeedbackHorizonResult> d_feedback(1);
  DeviceBuffer<u32> d_count(1);
  upload(d_candidates, candidate_handles);
  upload(d_candidate_distances, candidate_distances);
  upload(d_beam, old_handles);
  upload(d_ids, old_ids);
  upload(d_distances, old_distances);
  upload(d_expanded, old_expanded);
  feedback_merge_kernel<<<1, threads>>>(
    d_candidates.get(), d_candidate_distances.get(),
    static_cast<u32>(candidate_handles.size()), d_beam.get(), d_ids.get(),
    d_distances.get(), d_expanded.get(),
    old_count, beam_capacity,
    d_scratch_handles.get(), d_scratch_ids.get(),
    d_scratch_distances.get(), d_scratch_expanded.get(),
    d_compact_handles.get(), d_compact_flags.get(),
    d_compact_distances.get(), d_feedback.get(), feedback_cap, d_count.get());
  check_cuda(cudaGetLastError(), "feedback_merge_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "feedback_merge_kernel");
  const auto output_count = download(d_count);
  const auto output_handles = download(d_beam);
  const auto output_distances = download(d_distances);
  const auto output_expanded = download(d_expanded);
  const auto output_ids = download(d_ids);
  const auto output_feedback = download(d_feedback);
  if (output_count[0] != expected.beam.size()) {
    throw std::runtime_error("feedback merge valid count mismatch");
  }
  for (u32 index = 0; index < output_count[0]; ++index) {
    if (output_handles[index] != expected.beam[index].handle ||
        output_distances[index] != expected.beam[index].distance ||
        output_expanded[index] != expected.beam[index].expanded ||
        output_ids[index] != UINT32_MAX) {
      throw std::runtime_error("feedback merge changed Beam semantics");
    }
  }
  const FeedbackHorizonResult& actual = output_feedback[0];
  if (actual.horizon != expected.feedback.horizon ||
      actual.earliest_new_output !=
        expected.feedback.earliest_new_output ||
      actual.old_unexpanded_before_new !=
        expected.feedback.old_unexpanded_before_new ||
      actual.new_candidates_in_beam !=
        expected.feedback.new_candidates_in_beam) {
    throw std::runtime_error("feedback horizon metadata mismatch");
  }
}

void run_matrix(u32 threads, u32 capacity) {
  std::vector<u64> handles(capacity);
  std::vector<f32> distances(capacity);
  std::vector<u8> expanded(capacity, 0);
  for (u32 index = 0; index < capacity; ++index) {
    handles[index] = 0x100000000ULL + index;
    distances[index] = static_cast<f32>(index);
  }

  run_case(threads, capacity, handles, distances, expanded,
           {0x200000001ULL}, {-1.0f});

  expanded[0] = 1;
  run_case(threads, capacity, handles, distances, expanded,
           {0x200000002ULL}, {2.5f});

  run_case(threads, capacity, handles, distances, expanded,
           {0x200000003ULL}, {10000.0f});

  run_case(threads, capacity, handles, distances, expanded,
           {0x200000004ULL, 0x200000005ULL, 0x200000006ULL},
           {1.5f, -2.0f, 4.5f});

  // Stable radix ordering keeps old Beam entries before an equal-distance
  // candidate, regardless of the candidate handle value.
  run_case(threads, capacity, handles, distances, expanded,
           {1ULL}, {2.0f});

  run_case(threads, capacity, handles, distances, expanded,
           {gpu_search::kInvalidDeviceHandle, 0x200000007ULL},
           {-100.0f, std::numeric_limits<f32>::infinity()});

  std::vector<u64> short_handles(handles.begin(), handles.begin() + 5);
  std::vector<f32> short_distances(
    distances.begin(), distances.begin() + 5);
  std::vector<u8> short_expanded(expanded.begin(), expanded.begin() + 5);
  run_case(threads, capacity, short_handles, short_distances, short_expanded,
           {0x200000008ULL}, {1.5f});
}

}  // namespace

int main() {
  try {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
      std::cout << "SKIP: no CUDA device available\n";
      return 0;
    }
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    run_matrix(128, 128);
    run_matrix(256, 128);
    // Exercises the compact final-256 path.
    run_matrix(128, 256);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
