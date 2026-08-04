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
using gpu_search::BeamMergePolicy;
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
                          std::max<size_t>(count, 1) * sizeof(T)), "cudaMalloc");
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
  if (source.empty()) return;
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

__global__ void beam_merge_kernel(
    u64* candidate_handles, f32* candidate_distances, u32 candidate_count,
    u64* beam_handles, u32* beam_ids, f32* beam_distances,
    u8* beam_expanded, u32 initial_beam_count, u32 beam_capacity,
    u64* scratch_handles, u8* scratch_flags, f32* scratch_distances,
    BeamMergePolicy policy, BeamMergeCycleBreakdown* cycle_breakdown,
    u32* output_count) {
  __shared__ CandidateWorkspace workspace;
  __shared__ u32 beam_count;
  if (threadIdx.x == 0) beam_count = initial_beam_count;
  __syncthreads();
  merge_approximate_into_beam(
    candidate_handles, candidate_distances, candidate_count,
    beam_handles, beam_ids, beam_distances, beam_expanded,
    beam_count, beam_capacity, scratch_handles, scratch_flags,
    scratch_distances, workspace, policy, cycle_breakdown);
  if (threadIdx.x == 0) *output_count = beam_count;
}

struct HostItem {
  u64 handle;
  f32 distance;
  u8 expanded;
};

struct HostReference {
  std::vector<HostItem> beam;
};

HostReference reference_merge(
    const std::vector<u64>& old_handles,
    const std::vector<f32>& old_distances,
    const std::vector<u8>& old_expanded,
    const std::vector<u64>& candidate_handles,
    const std::vector<f32>& candidate_distances,
    u32 beam_capacity) {
  std::vector<HostItem> items;
  for (u32 index = 0; index < old_handles.size(); ++index) {
    items.push_back({
      old_handles[index], old_distances[index], old_expanded[index]});
  }
  for (u32 index = 0; index < candidate_handles.size(); ++index) {
    u64 handle = candidate_handles[index];
    f32 distance = candidate_distances[index];
    if (handle == gpu_search::kInvalidDeviceHandle ||
        !std::isfinite(distance)) {
      handle = gpu_search::kInvalidDeviceHandle;
      distance = FLT_MAX;
    }
    u8 expanded = 0;
    for (u32 prior = 0; prior < old_handles.size(); ++prior) {
      if (old_handles[prior] == handle) {
        expanded = old_expanded[prior];
        break;
      }
    }
    items.push_back({handle, distance, expanded});
  }
  std::stable_sort(items.begin(), items.end(),
    [](const HostItem& lhs, const HostItem& rhs) {
      return lhs.distance < rhs.distance;
    });
  HostReference result;
  for (const HostItem& item : items) {
    if (result.beam.size() == beam_capacity) break;
    if (item.handle == gpu_search::kInvalidDeviceHandle ||
        !std::isfinite(item.distance) || item.distance == FLT_MAX) {
      break;
    }
    result.beam.push_back(item);
  }
  return result;
}

void run_case(
    const std::string& case_name, u32 threads, u32 beam_capacity,
    std::vector<u64> old_handles, std::vector<f32> old_distances,
    std::vector<u8> old_expanded,
    std::vector<u64> candidate_handles,
    std::vector<f32> candidate_distances) {
  const u32 old_count = static_cast<u32>(old_handles.size());
  const HostReference expected = reference_merge(
    old_handles, old_distances, old_expanded,
    candidate_handles, candidate_distances, beam_capacity);
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
  DeviceBuffer<u64> d_compact_handles(beam_capacity * 4);
  DeviceBuffer<u8> d_compact_flags(beam_capacity * 4);
  DeviceBuffer<f32> d_compact_distances(beam_capacity * 4);
  DeviceBuffer<BeamMergeCycleBreakdown> d_cycle_breakdown(1);
  DeviceBuffer<u32> d_count(1);
  upload(d_candidates, candidate_handles);
  upload(d_candidate_distances, candidate_distances);
  for (const BeamMergePolicy policy :
       {BeamMergePolicy::legacy, BeamMergePolicy::stable_run}) {
    const std::string policy_name =
      policy == BeamMergePolicy::legacy ? "legacy" : "stable-run";
    const std::string label = case_name + "," + policy_name;
    upload(d_beam, old_handles);
    upload(d_ids, old_ids);
    upload(d_distances, old_distances);
    upload(d_expanded, old_expanded);
    check_cuda(cudaMemset(d_cycle_breakdown.get(), 0,
                          sizeof(BeamMergeCycleBreakdown)),
               "cudaMemset cycle breakdown");
    beam_merge_kernel<<<1, threads>>>(
      d_candidates.get(), d_candidate_distances.get(),
      static_cast<u32>(candidate_handles.size()), d_beam.get(), d_ids.get(),
      d_distances.get(), d_expanded.get(),
      old_count, beam_capacity,
      d_compact_handles.get(), d_compact_flags.get(),
      d_compact_distances.get(), policy,
      d_cycle_breakdown.get(), d_count.get());
    check_cuda(cudaGetLastError(), "beam_merge_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "beam_merge_kernel");
    const auto output_count = download(d_count);
    const auto output_handles = download(d_beam);
    const auto output_distances = download(d_distances);
    const auto output_expanded = download(d_expanded);
    const auto output_ids = download(d_ids);
    const auto cycle_breakdown = download(d_cycle_breakdown);
    if (output_count[0] != expected.beam.size()) {
      throw std::runtime_error(
        label + ": Beam merge valid count mismatch: expected " +
        std::to_string(expected.beam.size()) + ", got " +
        std::to_string(output_count[0]));
    }
    for (u32 index = 0; index < output_count[0]; ++index) {
      if (output_handles[index] != expected.beam[index].handle ||
          output_distances[index] != expected.beam[index].distance ||
          output_expanded[index] != expected.beam[index].expanded ||
          output_ids[index] != UINT32_MAX) {
        throw std::runtime_error(
          label + ": Beam merge changed semantics at output " +
          std::to_string(index));
      }
    }
    if (policy == BeamMergePolicy::stable_run &&
        (cycle_breakdown[0].sort == 0 ||
         cycle_breakdown[0].materialize == 0)) {
      throw std::runtime_error(
        label + ": stable-run cycle breakdown was not populated");
    }
  }
}

void run_matrix(u32 threads, u32 capacity) {
  const std::string prefix = "threads=" + std::to_string(threads) +
    ",capacity=" + std::to_string(capacity) + ",";
  std::vector<u64> handles(capacity);
  std::vector<f32> distances(capacity);
  std::vector<u8> expanded(capacity, 0);
  for (u32 index = 0; index < capacity; ++index) {
    handles[index] = 0x100000000ULL + index;
    distances[index] = static_cast<f32>(index);
  }

  run_case(prefix + "new-first", threads, capacity,
           handles, distances, expanded,
           {0x200000001ULL}, {-1.0f});

  expanded[0] = 1;
  run_case(prefix + "new-middle-expanded-prefix", threads, capacity,
           handles, distances, expanded,
           {0x200000002ULL}, {2.5f});

  run_case(prefix + "new-truncated", threads, capacity,
           handles, distances, expanded,
           {0x200000003ULL}, {10000.0f});

  run_case(prefix + "multiple-new", threads, capacity,
           handles, distances, expanded,
           {0x200000004ULL, 0x200000005ULL, 0x200000006ULL},
           {1.5f, -2.0f, 4.5f});

  // Stable radix ordering keeps old Beam entries before an equal-distance
  // candidate, regardless of the candidate handle value.
  run_case(prefix + "stable-tie-old-before-smaller-handle", threads, capacity,
           handles, distances, expanded,
           {1ULL}, {2.0f});

  run_case(prefix + "stable-tie-candidate-input-order", threads, capacity,
           handles, distances, expanded,
           {0xf00000001ULL, 1ULL, 0xa00000001ULL},
           {-5.0f, -5.0f, -5.0f});

  // CUB treats -0 and +0 as equivalent radix keys.  The existing +0 Beam
  // entry must therefore remain first, followed by candidate input order.
  run_case(prefix + "stable-tie-signed-zero", threads, capacity,
           handles, distances, expanded,
           {0x20000000bULL, 0x20000000cULL}, {-0.0f, 0.0f});

  // A candidate whose handle already exists in the authoritative Beam is
  // classified as old and inherits that entry's expanded bit, independent of
  // the candidate's distance or input position.
  expanded[3] = 1;
  run_case(prefix + "candidate-equals-expanded-old", threads, capacity,
           handles, distances, expanded,
           {handles[3], handles[5], handles[7]}, {-3.0f, 5.5f, 7.0f});

  run_case(prefix + "invalid-and-nonfinite", threads, capacity,
           handles, distances, expanded,
           {gpu_search::kInvalidDeviceHandle, 0x200000007ULL,
            0x200000008ULL, 0x200000009ULL, 0x20000000aULL},
           {-100.0f, std::numeric_limits<f32>::infinity(),
            -std::numeric_limits<f32>::infinity(),
            std::numeric_limits<f32>::quiet_NaN(), FLT_MAX});

  std::vector<u64> short_handles(handles.begin(), handles.begin() + 5);
  std::vector<f32> short_distances(
    distances.begin(), distances.begin() + 5);
  std::vector<u8> short_expanded(expanded.begin(), expanded.begin() + 5);
  run_case(prefix + "beam-not-full", threads, capacity,
           short_handles, short_distances, short_expanded,
           {0x200000008ULL}, {1.5f});

  // Stress stable ordering at both the Beam truncation boundary and the
  // compact implementation's 1024-item pass boundary.  Every equal-distance
  // old item must remain ahead of every candidate even when candidate handles
  // are numerically smaller.
  const u32 maximum_candidates =
    gpu_search::kPersistentMaxMergeCandidates - capacity;
  std::vector<u64> tied_candidates(maximum_candidates);
  std::vector<f32> tied_distances(maximum_candidates, 7.0f);
  std::vector<f32> tied_old_distances(capacity, 7.0f);
  for (u32 index = 0; index < maximum_candidates; ++index) {
    tied_candidates[index] = static_cast<u64>(index + 1);
  }
  run_case(prefix + "full-equal-run-topk-boundary", threads, capacity,
           handles, tied_old_distances, expanded,
           tied_candidates, tied_distances);

  // Place the globally best candidate immediately before and after the
  // compact 1024-item pass boundary.  This is also a wide-path reference
  // case, so both implementations are required to produce the same Beam.
  std::vector<u64> boundary_candidates(maximum_candidates);
  std::vector<f32> boundary_distances(maximum_candidates, 5000.0f);
  for (u32 index = 0; index < maximum_candidates; ++index) {
    boundary_candidates[index] = 0x300000000ULL + index;
    boundary_distances[index] = 5000.0f + static_cast<f32>(index);
  }
  const u32 pass_items =
    kApproximateSortThreadsCompact * kApproximateSortItemsCompactPass;
  if (pass_items > capacity &&
      pass_items - capacity < maximum_candidates) {
    const u32 before = pass_items - capacity - 1;
    const u32 after = pass_items - capacity;
    boundary_distances[before] = -4.0f;
    boundary_distances[after] = -3.0f;
  }
  run_case(prefix + "compact-pass-boundary", threads, capacity,
           handles, distances, expanded,
           boundary_candidates, boundary_distances);

  // The 128-thread stable-run path folds candidates [0, 1024) first and
  // candidates [1024, ...) second.  Force both sides of that continuation
  // boundary into the final Beam so the second fold and its origin metadata
  // are compared directly with the legacy path and CPU reference.
  constexpr u32 stable_fold_boundary = 1024;
  if (maximum_candidates > stable_fold_boundary) {
    std::fill(
      boundary_distances.begin(), boundary_distances.end(), 5000.0f);
    boundary_distances[stable_fold_boundary - 1] = -6.0f;
    boundary_distances[stable_fold_boundary] = -5.0f;
    run_case(prefix + "stable-second-fold-boundary", threads, capacity,
             handles, distances, expanded,
             boundary_candidates, boundary_distances);
  }
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
    // Exercise both compact-final-256 and wide Beam=256 paths.
    run_matrix(128, 256);
    run_matrix(256, 256);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
