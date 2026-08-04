#include <cuda_runtime.h>
#include <cub/warp/warp_merge_sort.cuh>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_search/persistent_kernel/candidate_scoring.cuh"

namespace {

using gpu_search::f32;
using gpu_search::kInvalidDeviceHandle;
using gpu_search::kPersistentFrontierRobCapacity;
using gpu_search::kPersistentMaxBeam;
using gpu_search::kPersistentMaxMergeCandidates;
using gpu_search::u8;
using gpu_search::u16;
using gpu_search::u32;
using gpu_search::u64;
using namespace gpu_search::persistent_kernel_detail;

constexpr u32 kBeamCapacity = kPersistentMaxBeam;
constexpr u32 kCandidateCapacity = kPersistentMaxMergeCandidates;
constexpr u32 kCandidateRunCount = 4;
constexpr u32 kCycleWarmupIterations = 16;
constexpr u32 kCycleMeasuredIterations = 256;

union CycleAlgorithmWorkspace {
  CandidateWorkspace stable_run;
  ApproximateWarpLeafSortStorage leaf_sorts;
};

struct CaseSpec {
  u32 candidate_count{};
  u32 beam_capacity{};
  u32 beam_count{};
  u32 issue_capacity{};
  u32 mode{};
};

struct CaseResult {
  u32 mismatch{};
  u32 first_mismatch{UINT32_MAX};
  u32 certificate_count{};
  u32 warp_merge_count{};
  u32 serial_count{};
  u32 tree_count{};
};

struct CycleResult {
  u64 pbec_cycles{};
  u64 warp_merge_cycles{};
  u64 full_prepare_preview_cycles{};
  u64 reusable_prepare_certificate_cycles{};
  u32 pbec_count{};
  u32 warp_merge_count{};
  u32 full_count{};
  u32 reusable_count{};
  u32 mismatch{};
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
    check_cuda(cudaMalloc(
      reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc");
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }

  T* get() const {
    return data_;
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

 private:
  T* data_{};
};

template <typename T>
void upload(DeviceBuffer<T>& destination, const std::vector<T>& source) {
  check_cuda(cudaMemcpy(
    destination.get(), source.data(), source.size() * sizeof(T),
    cudaMemcpyHostToDevice), "cudaMemcpy H2D");
}

template <typename T>
std::vector<T> download(const DeviceBuffer<T>& source, size_t count) {
  std::vector<T> result(count);
  check_cuda(cudaMemcpy(
    result.data(), source.get(), count * sizeof(T),
    cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  return result;
}

__global__ void partition_bounded_certificate_equivalence_kernel(
    const CaseSpec* specs,
    const u64* input_beam_handles, const f32* input_beam_distances,
    const u8* input_beam_expanded,
    u64* candidate_handles, f32* candidate_distances,
    u64* beam_handles, u32* beam_ids, f32* beam_distances,
    u8* beam_expanded,
    u64* scratch_handles, u8* scratch_flags, f32* scratch_distances,
    CaseResult* results) {
  __shared__ CycleAlgorithmWorkspace algorithm_workspace;
  __shared__ StableMergePreparedState state;
  __shared__ BeamMergeCycleBreakdown phases;
  __shared__ u32 beam_count;
  __shared__ u32 certificate_count;
  __shared__ u32 warp_merge_count;
  __shared__ u32 serial_count;
  __shared__ u32 tree_count;
  __shared__ u32 mismatch;
  __shared__ u32 first_mismatch;
  __shared__ u64 certificate_handles[kPersistentFrontierRobCapacity];
  __shared__ u16 certificate_ranks[kPersistentFrontierRobCapacity];
  __shared__ u64 serial_handles[kPersistentFrontierRobCapacity];
  __shared__ u16 serial_ranks[kPersistentFrontierRobCapacity];
  __shared__ u64 warp_merge_handles[kPersistentFrontierRobCapacity];
  __shared__ u16 warp_merge_ranks[kPersistentFrontierRobCapacity];
  __shared__ u64 tree_handles[kPersistentFrontierRobCapacity];
  __shared__ u16 tree_ranks[kPersistentFrontierRobCapacity];
  __shared__ u64 warp_final_handles[kBeamCapacity];
  __shared__ u32 warp_final_ids[kBeamCapacity];
  __shared__ f32 warp_final_distances[kBeamCapacity];
  __shared__ u8 warp_final_expanded[kBeamCapacity];
  __shared__ u32 warp_final_count;
  CandidateWorkspace& workspace = algorithm_workspace.stable_run;

  const u32 case_index = blockIdx.x;
  const CaseSpec spec = specs[case_index];
  const size_t beam_offset =
    static_cast<size_t>(case_index) * kBeamCapacity;
  const size_t candidate_offset =
    static_cast<size_t>(case_index) * kCandidateCapacity;
  const size_t scratch_offset =
    static_cast<size_t>(case_index) *
    kCandidateRunCount * kBeamCapacity;
  const u64* original_handles = input_beam_handles + beam_offset;
  const f32* original_distances = input_beam_distances + beam_offset;
  const u8* original_expanded = input_beam_expanded + beam_offset;
  u64* case_candidates = candidate_handles + candidate_offset;
  f32* case_candidate_distances = candidate_distances + candidate_offset;
  u64* case_beam_handles = beam_handles + beam_offset;
  u32* case_beam_ids = beam_ids + beam_offset;
  f32* case_beam_distances = beam_distances + beam_offset;
  u8* case_beam_expanded = beam_expanded + beam_offset;
  u64* case_scratch_handles = scratch_handles + scratch_offset;
  u8* case_scratch_flags = scratch_flags + scratch_offset;
  f32* case_scratch_distances = scratch_distances + scratch_offset;

  for (u32 rank = threadIdx.x; rank < kBeamCapacity;
       rank += blockDim.x) {
    case_beam_handles[rank] = original_handles[rank];
    case_beam_ids[rank] = rank;
    case_beam_distances[rank] = original_distances[rank];
    case_beam_expanded[rank] = original_expanded[rank];
  }
  if (threadIdx.x == 0) {
    beam_count = spec.beam_count;
    certificate_count = 0;
    warp_merge_count = 0;
    serial_count = 0;
    tree_count = 0;
    mismatch = 0;
    first_mismatch = UINT32_MAX;
    state = {};
    phases = {};
  }
  __syncthreads();

  prepare_partition_bounded_exact_certificate(
    case_candidates, case_candidate_distances, spec.candidate_count,
    case_beam_handles, case_beam_distances, case_beam_expanded,
    beam_count, spec.beam_capacity,
    case_scratch_handles, case_scratch_distances,
    workspace.arrays, spec.issue_capacity,
    certificate_handles, certificate_ranks, certificate_count);

  // A speculative certificate must not publish any authoritative Beam state.
  for (u32 rank = threadIdx.x; rank < kBeamCapacity;
       rank += blockDim.x) {
    if (case_beam_handles[rank] != original_handles[rank] ||
        __float_as_uint(case_beam_distances[rank]) !=
          __float_as_uint(original_distances[rank]) ||
        case_beam_expanded[rank] != original_expanded[rank] ||
        case_beam_ids[rank] != rank) {
      atomicOr(&mismatch, 1u);
      atomicMin(&first_mismatch, rank);
    }
  }
  __syncthreads();

  prepare_warp_leaf_fused_frontier_certificate(
    case_candidates, case_candidate_distances, spec.candidate_count,
    case_beam_handles, case_beam_distances, case_beam_expanded,
    beam_count, spec.beam_capacity,
    case_scratch_handles, case_scratch_flags, case_scratch_distances,
    algorithm_workspace.leaf_sorts,
    workspace.arrays, spec.issue_capacity,
    warp_merge_handles, warp_merge_ranks, warp_merge_count,
    state, &phases);

  const u32 active_candidate_runs = min(
    kCandidateRunCount,
    (spec.candidate_count + kWarpLeafSortCapacity - 1u) /
      kWarpLeafSortCapacity);

  // PFEC must leave the authoritative Beam byte-identical and prepare the
  // same restore=false state/scratch contract as Stable-Run.
  for (u32 rank = threadIdx.x; rank < kBeamCapacity;
       rank += blockDim.x) {
    if (case_beam_handles[rank] != original_handles[rank] ||
        __float_as_uint(case_beam_distances[rank]) !=
          __float_as_uint(original_distances[rank]) ||
        case_beam_expanded[rank] != original_expanded[rank] ||
        case_beam_ids[rank] != rank) {
      atomicOr(&mismatch, 1u);
      atomicMin(&first_mismatch, rank);
    }
  }
  for (u32 index = threadIdx.x;
       index < active_candidate_runs * spec.beam_capacity;
       index += blockDim.x) {
    if (case_scratch_flags[index] != 0) {
      atomicOr(&mismatch, 64u);
      atomicMin(&first_mismatch, index);
    }
  }
  if (threadIdx.x == 0 &&
      (state.original_count != spec.beam_count ||
       state.candidate_run_count != active_candidate_runs ||
       state.compact == 0 || state.origin_copied != 0 ||
       state.prepared == 0)) {
    mismatch |= 64u;
  }
  __syncthreads();

  finish_approximate_stable_runs(
    case_beam_handles, case_beam_ids, case_beam_distances,
    case_beam_expanded, beam_count, spec.beam_capacity,
    case_scratch_handles, case_scratch_flags, case_scratch_distances,
    workspace, state, &phases, true);
  for (u32 rank = threadIdx.x; rank < spec.beam_capacity;
       rank += blockDim.x) {
    warp_final_handles[rank] = case_beam_handles[rank];
    warp_final_ids[rank] = case_beam_ids[rank];
    warp_final_distances[rank] = case_beam_distances[rank];
    warp_final_expanded[rank] = case_beam_expanded[rank];
  }
  if (threadIdx.x == 0) warp_final_count = beam_count;
  __syncthreads();

  // Restore the immutable input and run the current BlockRadix Stable-Run as
  // the bitwise final-Beam oracle.
  for (u32 rank = threadIdx.x; rank < kBeamCapacity;
       rank += blockDim.x) {
    case_beam_handles[rank] = original_handles[rank];
    case_beam_ids[rank] = rank;
    case_beam_distances[rank] = original_distances[rank];
    case_beam_expanded[rank] = original_expanded[rank];
  }
  if (threadIdx.x == 0) beam_count = spec.beam_count;
  __syncthreads();

  // Build the complete four Stable-Run leaves after the PBEC snapshot.  This
  // is the unmodified authoritative pipeline and serves as the exact oracle.
  prepare_approximate_stable_runs(
    case_candidates, case_candidate_distances, spec.candidate_count,
    case_beam_handles, case_beam_ids, case_beam_distances,
    case_beam_expanded, beam_count, spec.beam_capacity,
    case_scratch_handles, case_scratch_flags, case_scratch_distances,
    workspace, state, &phases, false);

  preview_serial_stable_unexpanded_frontier(
    case_beam_handles, case_beam_distances, case_beam_expanded,
    beam_count, spec.beam_capacity,
    case_scratch_handles, case_scratch_flags, case_scratch_distances,
    state.candidate_run_count, spec.issue_capacity,
    serial_handles, serial_ranks, serial_count);

  preview_tree_stable_unexpanded_frontier(
    case_beam_handles, case_beam_distances, case_beam_expanded,
    beam_count, spec.beam_capacity,
    case_scratch_handles, case_scratch_distances,
    state.candidate_run_count, spec.issue_capacity, workspace.arrays,
    tree_handles, tree_ranks, tree_count);

  if (threadIdx.x == 0) {
    if (certificate_count != serial_count) mismatch |= 2u;
    if (certificate_count != tree_count) mismatch |= 4u;
    if (warp_merge_count != serial_count ||
        warp_merge_count != tree_count) {
      mismatch |= 32u;
    }
    if (certificate_count > spec.issue_capacity ||
        certificate_count > kPersistentFrontierRobCapacity) {
      mismatch |= 8u;
    }
  }
  __syncthreads();
  for (u32 output = threadIdx.x;
       output < kPersistentFrontierRobCapacity;
       output += blockDim.x) {
    if (output < certificate_count &&
        (output >= serial_count ||
         certificate_handles[output] != serial_handles[output] ||
         certificate_ranks[output] != serial_ranks[output])) {
      atomicOr(&mismatch, 2u);
      atomicMin(&first_mismatch, output);
    }
    if (output < warp_merge_count &&
        (output >= serial_count ||
         warp_merge_handles[output] != serial_handles[output] ||
         warp_merge_ranks[output] != serial_ranks[output])) {
      atomicOr(&mismatch, 32u);
      atomicMin(&first_mismatch, output);
    }
    if (output < certificate_count &&
        (output >= tree_count ||
         certificate_handles[output] != tree_handles[output] ||
         certificate_ranks[output] != tree_ranks[output])) {
      atomicOr(&mismatch, 4u);
      atomicMin(&first_mismatch, output);
    }
    if (output != 0 && output < certificate_count &&
        certificate_ranks[output] <= certificate_ranks[output - 1u]) {
      atomicOr(&mismatch, 8u);
      atomicMin(&first_mismatch, output);
    }
  }
  __syncthreads();

  // The final comparison is against the complete authoritative Beam, not
  // another preview implementation.  It also proves that PBEC's returned
  // ranks include skipped expanded old-Beam entries exactly.
  finish_approximate_stable_runs(
    case_beam_handles, case_beam_ids, case_beam_distances,
    case_beam_expanded, beam_count, spec.beam_capacity,
    case_scratch_handles, case_scratch_flags, case_scratch_distances,
    workspace, state, &phases, true);
  for (u32 rank = threadIdx.x; rank < spec.beam_capacity;
       rank += blockDim.x) {
    if (beam_count != warp_final_count ||
        case_beam_handles[rank] != warp_final_handles[rank] ||
        case_beam_ids[rank] != warp_final_ids[rank] ||
        __float_as_uint(case_beam_distances[rank]) !=
          __float_as_uint(warp_final_distances[rank]) ||
        case_beam_expanded[rank] != warp_final_expanded[rank]) {
      atomicOr(&mismatch, 128u);
      atomicMin(&first_mismatch, rank);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    u32 output = 0;
    for (u32 rank = 0;
         rank < beam_count && output < spec.issue_capacity; ++rank) {
      if (case_beam_expanded[rank] != 0) continue;
      if (output >= certificate_count ||
          certificate_handles[output] != case_beam_handles[rank] ||
          certificate_ranks[output] != rank) {
        mismatch |= 16u;
        first_mismatch = min(first_mismatch, output);
        break;
      }
      if (output >= warp_merge_count ||
          warp_merge_handles[output] != case_beam_handles[rank] ||
          warp_merge_ranks[output] != rank) {
        mismatch |= 32u;
        first_mismatch = min(first_mismatch, output);
        break;
      }
      ++output;
    }
    if (output != certificate_count) mismatch |= 16u;
    if (output != warp_merge_count) mismatch |= 32u;
    results[case_index] = CaseResult{
      mismatch, first_mismatch, certificate_count, warp_merge_count,
      serial_count, tree_count};
  }
}

__global__ void partition_bounded_certificate_cycle_kernel(
    const u64* candidate_handles, const f32* candidate_distances,
    u32 issue_capacity, CycleResult* result) {
  __shared__ CycleAlgorithmWorkspace algorithm_workspace;
  __shared__ StableMergePreparedState state;
  __shared__ u64 beam_handles[kBeamCapacity];
  __shared__ u32 beam_ids[kBeamCapacity];
  __shared__ f32 beam_distances[kBeamCapacity];
  __shared__ u8 beam_expanded[kBeamCapacity];
  __shared__ u64 scratch_handles[
    kCandidateRunCount * kBeamCapacity];
  __shared__ u8 scratch_flags[
    kCandidateRunCount * kBeamCapacity];
  __shared__ f32 scratch_distances[
    kCandidateRunCount * kBeamCapacity];
  __shared__ u64 pbec_handles[kPersistentFrontierRobCapacity];
  __shared__ u16 pbec_ranks[kPersistentFrontierRobCapacity];
  __shared__ u64 warp_merge_handles[kPersistentFrontierRobCapacity];
  __shared__ u16 warp_merge_ranks[kPersistentFrontierRobCapacity];
  __shared__ u64 full_handles[kPersistentFrontierRobCapacity];
  __shared__ u16 full_ranks[kPersistentFrontierRobCapacity];
  __shared__ u64 reusable_handles[kPersistentFrontierRobCapacity];
  __shared__ u16 reusable_ranks[kPersistentFrontierRobCapacity];
  __shared__ u32 beam_count;
  __shared__ u32 pbec_count;
  __shared__ u32 warp_merge_count;
  __shared__ u32 full_count;
  __shared__ u32 reusable_count;
  __shared__ u32 mismatch;
  __shared__ u64 phase_started;
  __shared__ u64 pbec_cycles;
  __shared__ u64 warp_merge_cycles;
  __shared__ u64 full_cycles;
  __shared__ u64 reusable_cycles;
  CandidateWorkspace& workspace = algorithm_workspace.stable_run;

  for (u32 rank = threadIdx.x; rank < kBeamCapacity;
       rank += blockDim.x) {
    beam_handles[rank] = 0x100000000ULL + rank;
    beam_ids[rank] = rank;
    beam_distances[rank] = static_cast<f32>(rank / 3u) * 0.375f;
    beam_expanded[rank] = static_cast<u8>((rank % 9u) == 0u);
  }
  if (threadIdx.x == 0) {
    beam_count = kBeamCapacity;
    pbec_count = 0;
    warp_merge_count = 0;
    full_count = 0;
    reusable_count = 0;
    mismatch = 0;
    phase_started = 0;
    pbec_cycles = 0;
    warp_merge_cycles = 0;
    full_cycles = 0;
    reusable_cycles = 0;
    state = {};
  }
  __syncthreads();

  for (u32 iteration = 0;
       iteration < kCycleWarmupIterations + kCycleMeasuredIterations;
       ++iteration) {
    if (threadIdx.x == 0) {
      pbec_count = 0;
      warp_merge_count = 0;
      full_count = 0;
      reusable_count = 0;
      phase_started = clock64();
    }
    __syncthreads();
    prepare_partition_bounded_exact_certificate(
      candidate_handles, candidate_distances, 1536u,
      beam_handles, beam_distances, beam_expanded,
      beam_count, kBeamCapacity,
      scratch_handles, scratch_distances,
      workspace.arrays, issue_capacity,
      pbec_handles, pbec_ranks, pbec_count);
    if (threadIdx.x == 0 && iteration >= kCycleWarmupIterations) {
      pbec_cycles += clock64() - phase_started;
    }
    __syncthreads();

    if (threadIdx.x == 0) phase_started = clock64();
    __syncthreads();
    prepare_warp_leaf_fused_frontier_certificate(
      candidate_handles, candidate_distances, 1536u,
      beam_handles, beam_distances, beam_expanded,
      beam_count, kBeamCapacity,
      scratch_handles, scratch_flags, scratch_distances,
      algorithm_workspace.leaf_sorts,
      workspace.arrays, issue_capacity,
      warp_merge_handles, warp_merge_ranks, warp_merge_count,
      state);
    if (threadIdx.x == 0 && iteration >= kCycleWarmupIterations) {
      warp_merge_cycles += clock64() - phase_started;
    }
    __syncthreads();

    if (threadIdx.x == 0) phase_started = clock64();
    __syncthreads();
    prepare_approximate_stable_runs(
      const_cast<u64*>(candidate_handles),
      const_cast<f32*>(candidate_distances), 1536u,
      beam_handles, beam_ids, beam_distances, beam_expanded,
      beam_count, kBeamCapacity,
      scratch_handles, scratch_flags, scratch_distances,
      workspace, state, nullptr, false);
    preview_tree_stable_unexpanded_frontier(
      beam_handles, beam_distances, beam_expanded,
      beam_count, kBeamCapacity,
      scratch_handles, scratch_distances,
      state.candidate_run_count, issue_capacity,
      workspace.arrays, full_handles, full_ranks, full_count);
    if (threadIdx.x == 0 && iteration >= kCycleWarmupIterations) {
      full_cycles += clock64() - phase_started;
    }
    __syncthreads();

    if (threadIdx.x == 0) phase_started = clock64();
    __syncthreads();
    prepare_approximate_stable_runs(
      const_cast<u64*>(candidate_handles),
      const_cast<f32*>(candidate_distances), 1536u,
      beam_handles, beam_ids, beam_distances, beam_expanded,
      beam_count, kBeamCapacity,
      scratch_handles, scratch_flags, scratch_distances,
      workspace, state, nullptr, false);
    prepare_reusable_fused_frontier_certificate(
      beam_handles, beam_distances, beam_expanded,
      kBeamCapacity, beam_count,
      scratch_handles, scratch_flags, scratch_distances,
      state.candidate_run_count, workspace.arrays, issue_capacity,
      reusable_handles, reusable_ranks, reusable_count, state);
    if (threadIdx.x == 0 && iteration >= kCycleWarmupIterations) {
      reusable_cycles += clock64() - phase_started;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && pbec_count != full_count) {
    mismatch = 1;
  }
  if (threadIdx.x == 0 && warp_merge_count != full_count) {
    mismatch = 1;
  }
  if (threadIdx.x == 0 && reusable_count != full_count) {
    mismatch = 1;
  }
  __syncthreads();
  for (u32 output = threadIdx.x;
       output < kPersistentFrontierRobCapacity;
       output += blockDim.x) {
    if (output < pbec_count &&
        (output >= full_count ||
         pbec_handles[output] != full_handles[output] ||
         pbec_ranks[output] != full_ranks[output])) {
      atomicExch(&mismatch, 1u);
    }
    if (output < warp_merge_count &&
        (output >= full_count ||
         warp_merge_handles[output] != full_handles[output] ||
         warp_merge_ranks[output] != full_ranks[output])) {
      atomicExch(&mismatch, 1u);
    }
    if (output < reusable_count &&
        (output >= full_count ||
         reusable_handles[output] != full_handles[output] ||
         reusable_ranks[output] != full_ranks[output])) {
      atomicExch(&mismatch, 1u);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    *result = CycleResult{
      pbec_cycles, warp_merge_cycles, full_cycles, reusable_cycles,
      pbec_count, warp_merge_count, full_count, reusable_count, mismatch};
  }
}

std::vector<CaseSpec> make_specs() {
  constexpr u32 candidate_counts[]{
    0u, 1u, 31u, 32u, 33u, 127u, 128u,
    511u, 512u, 513u, 777u, 1023u, 1024u, 1025u,
    1535u, 1536u, 1537u, 2047u, 2048u};
  constexpr u32 issue_capacities[]{1u, 8u, 16u, 17u, 31u, 32u};
  constexpr u32 beam_capacities[]{64u, 128u};
  std::vector<CaseSpec> specs;
  specs.reserve(
    std::size(candidate_counts) * std::size(issue_capacities) *
    std::size(beam_capacities));
  u32 ordinal = 0;
  for (const u32 beam_capacity : beam_capacities) {
    for (const u32 candidate_count : candidate_counts) {
      for (const u32 issue_capacity : issue_capacities) {
        const u32 count_selector = ordinal % 6u;
        const u32 beam_count = count_selector == 0u ? 0u :
          count_selector == 1u ? 1u :
          count_selector == 2u ? beam_capacity / 3u :
          count_selector == 3u ? beam_capacity / 2u :
          count_selector == 4u ? beam_capacity - 1u :
          beam_capacity;
        specs.push_back(CaseSpec{
          candidate_count, beam_capacity, beam_count,
          issue_capacity, ordinal % 7u});
        ++ordinal;
      }
    }
  }
  return specs;
}

f32 candidate_distance(u32 mode, u32 case_index, u32 ordinal) {
  const u32 leaf_ordinal = ordinal % 512u;
  switch (mode) {
    case 0:
      return 1.0f;
    case 1:
      return (ordinal & 1u) == 0 ? -0.0f : 0.0f;
    case 2:
      return static_cast<f32>(
        (ordinal * 37u + case_index * 13u + 11u) % 257u) * 0.125f;
    case 3:
      return static_cast<f32>(511u - leaf_ordinal) * 0.0625f;
    case 4:
      return static_cast<f32>(leaf_ordinal / 16u) * 0.25f;
    case 5:
      return static_cast<f32>(
        (ordinal * 73u + case_index * 29u + 7u) % 89u) * 0.5f;
    default: {
      const std::int32_t centered = static_cast<std::int32_t>(
        (ordinal * 61u + case_index * 19u + 3u) % 127u) - 63;
      return static_cast<f32>(centered) * 0.25f;
    }
  }
}

void fill_inputs(
    const std::vector<CaseSpec>& specs,
    std::vector<u64>& beam_handles,
    std::vector<f32>& beam_distances,
    std::vector<u8>& beam_expanded,
    std::vector<u64>& candidate_handles,
    std::vector<f32>& candidate_distances) {
  for (u32 case_index = 0; case_index < specs.size(); ++case_index) {
    const CaseSpec spec = specs[case_index];
    const size_t beam_offset =
      static_cast<size_t>(case_index) * kBeamCapacity;
    const size_t candidate_offset =
      static_cast<size_t>(case_index) * kCandidateCapacity;
    for (u32 rank = 0; rank < kBeamCapacity; ++rank) {
      if (rank >= spec.beam_count) {
        beam_handles[beam_offset + rank] = kInvalidDeviceHandle;
        beam_distances[beam_offset + rank] = FLT_MAX;
        beam_expanded[beam_offset + rank] = 0;
        continue;
      }
      beam_handles[beam_offset + rank] =
        (u64{case_index + 1u} << 48u) | u64{rank + 1u};
      if (spec.mode == 0u) {
        beam_distances[beam_offset + rank] = 1.0f;
      } else if (spec.mode == 1u) {
        beam_distances[beam_offset + rank] =
          (rank & 1u) == 0 ? -0.0f : 0.0f;
      } else if (spec.mode == 6u) {
        const std::int32_t centered =
          static_cast<std::int32_t>(rank) -
          static_cast<std::int32_t>(spec.beam_capacity / 2u);
        beam_distances[beam_offset + rank] =
          static_cast<f32>(centered) * 0.125f;
      } else {
        beam_distances[beam_offset + rank] =
          static_cast<f32>(rank / 5u) * 0.125f;
      }
      if (spec.mode == 5) {
        beam_expanded[beam_offset + rank] =
          static_cast<u8>((rank % 11u) != 10u);
      } else {
        beam_expanded[beam_offset + rank] =
          static_cast<u8>(
            ((rank * 7u + case_index) % (spec.mode + 3u)) == 0u);
      }
    }

    for (u32 ordinal = 0; ordinal < kCandidateCapacity; ++ordinal) {
      u64 handle = kInvalidDeviceHandle;
      f32 distance = FLT_MAX;
      if (ordinal < spec.candidate_count) {
        handle =
          (u64{case_index + 1u} << 32u) | u64{ordinal + 1u};
        distance =
          candidate_distance(spec.mode, case_index, ordinal);

        const bool invalid_heavy =
          spec.mode == 5u && (ordinal % 3u) == 0u;
        const bool invalid_sparse =
          ((ordinal + case_index * 17u) % 97u) == 0u;
        if (invalid_heavy || invalid_sparse) {
          switch ((ordinal + case_index) & 3u) {
            case 0:
              handle = kInvalidDeviceHandle;
              break;
            case 1:
              distance = std::numeric_limits<f32>::quiet_NaN();
              break;
            case 2:
              distance = std::numeric_limits<f32>::infinity();
              break;
            default:
              distance = FLT_MAX;
              break;
          }
        }
      }
      candidate_handles[candidate_offset + ordinal] = handle;
      candidate_distances[candidate_offset + ordinal] = distance;
    }
  }
}

void run_equivalence_cases() {
  const std::vector<CaseSpec> specs = make_specs();
  const size_t beam_items =
    specs.size() * static_cast<size_t>(kBeamCapacity);
  const size_t candidate_items =
    specs.size() * static_cast<size_t>(kCandidateCapacity);
  const size_t scratch_items =
    specs.size() * static_cast<size_t>(
      kCandidateRunCount * kBeamCapacity);

  std::vector<u64> beam_handles(beam_items);
  std::vector<f32> beam_distances(beam_items);
  std::vector<u8> beam_expanded(beam_items);
  std::vector<u64> candidate_handles(candidate_items);
  std::vector<f32> candidate_distances(candidate_items);
  fill_inputs(
    specs, beam_handles, beam_distances, beam_expanded,
    candidate_handles, candidate_distances);

  DeviceBuffer<CaseSpec> d_specs(specs.size());
  DeviceBuffer<u64> d_input_beam_handles(beam_items);
  DeviceBuffer<f32> d_input_beam_distances(beam_items);
  DeviceBuffer<u8> d_input_beam_expanded(beam_items);
  DeviceBuffer<u64> d_candidate_handles(candidate_items);
  DeviceBuffer<f32> d_candidate_distances(candidate_items);
  DeviceBuffer<u64> d_beam_handles(beam_items);
  DeviceBuffer<u32> d_beam_ids(beam_items);
  DeviceBuffer<f32> d_beam_distances(beam_items);
  DeviceBuffer<u8> d_beam_expanded(beam_items);
  DeviceBuffer<u64> d_scratch_handles(scratch_items);
  DeviceBuffer<u8> d_scratch_flags(scratch_items);
  DeviceBuffer<f32> d_scratch_distances(scratch_items);
  DeviceBuffer<CaseResult> d_results(specs.size());
  upload(d_specs, specs);
  upload(d_input_beam_handles, beam_handles);
  upload(d_input_beam_distances, beam_distances);
  upload(d_input_beam_expanded, beam_expanded);
  upload(d_candidate_handles, candidate_handles);
  upload(d_candidate_distances, candidate_distances);

  partition_bounded_certificate_equivalence_kernel
    <<<static_cast<u32>(specs.size()), 128>>>(
      d_specs.get(),
      d_input_beam_handles.get(), d_input_beam_distances.get(),
      d_input_beam_expanded.get(),
      d_candidate_handles.get(), d_candidate_distances.get(),
      d_beam_handles.get(), d_beam_ids.get(), d_beam_distances.get(),
      d_beam_expanded.get(),
      d_scratch_handles.get(), d_scratch_flags.get(),
      d_scratch_distances.get(), d_results.get());
  check_cuda(
    cudaGetLastError(),
    "partition_bounded_certificate_equivalence_kernel launch");
  check_cuda(
    cudaDeviceSynchronize(),
    "partition_bounded_certificate_equivalence_kernel synchronize");

  const auto results = download(d_results, specs.size());
  for (u32 case_index = 0; case_index < specs.size(); ++case_index) {
    const CaseResult result = results[case_index];
    if (result.mismatch == 0) continue;
    const CaseSpec spec = specs[case_index];
    throw std::runtime_error(
      "PBEC case " + std::to_string(case_index) +
      " diverged: mask=" + std::to_string(result.mismatch) +
      ", first=" + std::to_string(result.first_mismatch) +
      ", candidates=" + std::to_string(spec.candidate_count) +
      ", capacity=" + std::to_string(spec.beam_capacity) +
      ", beam=" + std::to_string(spec.beam_count) +
      ", issue=" + std::to_string(spec.issue_capacity) +
      ", mode=" + std::to_string(spec.mode) +
      ", certificate=" + std::to_string(result.certificate_count) +
      ", warp_merge=" + std::to_string(result.warp_merge_count) +
      ", serial=" + std::to_string(result.serial_count) +
      ", tree=" + std::to_string(result.tree_count));
  }
  std::cout
    << "PASS: " << specs.size()
    << " PBEC K=64/128 boundary/tie/finite-negative/invalid/expanded cases "
       "and production PFEC match serial preview, tree preview, and the "
       "bitwise fully materialized authoritative Beam\n";
}

void run_cycle_microbenchmark(double cycles_per_microsecond) {
  constexpr u32 candidate_count = 1536;
  std::vector<u64> candidate_handles(candidate_count);
  std::vector<f32> candidate_distances(candidate_count);
  for (u32 ordinal = 0; ordinal < candidate_count; ++ordinal) {
    candidate_handles[ordinal] = 0x200000000ULL + ordinal;
    candidate_distances[ordinal] = static_cast<f32>(
      (ordinal * 37u + 11u) % 1024u) * 0.25f;
  }

  DeviceBuffer<u64> d_candidate_handles(candidate_count);
  DeviceBuffer<f32> d_candidate_distances(candidate_count);
  DeviceBuffer<CycleResult> d_result(1);
  upload(d_candidate_handles, candidate_handles);
  upload(d_candidate_distances, candidate_distances);

  for (const u32 issue_capacity : {16u, 32u}) {
    partition_bounded_certificate_cycle_kernel<<<1, 128>>>(
      d_candidate_handles.get(), d_candidate_distances.get(),
      issue_capacity, d_result.get());
    check_cuda(
      cudaGetLastError(),
      "partition_bounded_certificate_cycle_kernel launch");
    check_cuda(
      cudaDeviceSynchronize(),
      "partition_bounded_certificate_cycle_kernel synchronize");
    const CycleResult result = download(d_result, 1)[0];
    if (result.mismatch != 0 ||
        result.pbec_count != issue_capacity ||
        result.warp_merge_count != issue_capacity ||
        result.full_count != issue_capacity ||
        result.reusable_count != issue_capacity ||
        result.pbec_cycles == 0 ||
        result.warp_merge_cycles == 0 ||
        result.full_prepare_preview_cycles == 0 ||
        result.reusable_prepare_certificate_cycles == 0) {
      throw std::runtime_error(
        "PBEC cycle microbenchmark output mismatch at issue=" +
        std::to_string(issue_capacity));
    }

    const double divisor =
      static_cast<double>(kCycleMeasuredIterations);
    const double pbec_cycles =
      static_cast<double>(result.pbec_cycles) / divisor;
    const double warp_merge_cycles =
      static_cast<double>(result.warp_merge_cycles) / divisor;
    const double full_cycles =
      static_cast<double>(result.full_prepare_preview_cycles) / divisor;
    const double reusable_cycles = static_cast<double>(
      result.reusable_prepare_certificate_cycles) / divisor;
    std::cout
      << "pbec_cycle_microbench"
      << " candidates=" << candidate_count
      << " beam=" << kBeamCapacity
      << " issue=" << issue_capacity
      << " iterations=" << kCycleMeasuredIterations
      << " pbec_cycles=" << std::fixed << std::setprecision(1)
      << pbec_cycles
      << " pbec_us~=" << std::setprecision(3)
      << pbec_cycles / cycles_per_microsecond
      << " pfec_cycles=" << std::setprecision(1)
      << warp_merge_cycles
      << " pfec_us~=" << std::setprecision(3)
      << warp_merge_cycles / cycles_per_microsecond
      << " full_prepare_preview_cycles=" << std::setprecision(1)
      << full_cycles
      << " full_prepare_preview_us~=" << std::setprecision(3)
      << full_cycles / cycles_per_microsecond
      << " reusable_prepare_certificate_cycles=" << std::setprecision(1)
      << reusable_cycles
      << " reusable_prepare_certificate_us~=" << std::setprecision(3)
      << reusable_cycles / cycles_per_microsecond
      << " full_over_pbec=" << std::setprecision(2)
      << full_cycles / pbec_cycles
      << " full_over_pfec=" << std::setprecision(2)
      << full_cycles / warp_merge_cycles
      << " reusable_over_full=" << std::setprecision(2)
      << reusable_cycles / full_cycles
      << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: no CUDA device available\n";
    return 0;
  }
  try {
    const bool cycle_only =
      argc == 2 && std::string(argv[1]) == "--cycle-only";
    if (argc > 2 || (argc == 2 && !cycle_only)) {
      throw std::invalid_argument(
        "usage: gpu_partition_bounded_certificate_test [--cycle-only]");
    }
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp properties{};
    check_cuda(
      cudaGetDeviceProperties(&properties, 0),
      "cudaGetDeviceProperties");
    constexpr size_t stable_workspace_bytes = sizeof(CandidateWorkspace);
    constexpr size_t pfec_sort_storage_bytes =
      sizeof(ApproximateWarpLeafSortStorage);
    constexpr size_t overlay_workspace_bytes =
      std::max(stable_workspace_bytes, pfec_sort_storage_bytes);
    std::cout
      << "pfec_workspace stable_bytes=" << stable_workspace_bytes
      << " sort_storage_bytes=" << pfec_sort_storage_bytes
      << " overlay_bytes=" << overlay_workspace_bytes
      << " delta_bytes="
      << overlay_workspace_bytes - stable_workspace_bytes
      << '\n';
    if (!cycle_only) run_equivalence_cases();
    run_cycle_microbenchmark(
      static_cast<double>(properties.clockRate) / 1000.0);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
