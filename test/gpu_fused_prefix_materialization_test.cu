#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_search/persistent_kernel/candidate_scoring.cuh"

namespace {

using gpu_search::f32;
using gpu_search::kInvalidDeviceHandle;
using gpu_search::kPersistentMaxBeam;
using gpu_search::u8;
using gpu_search::u16;
using gpu_search::u32;
using gpu_search::u64;
using namespace gpu_search::persistent_kernel_detail;

constexpr u32 kBeamCapacity = kPersistentMaxBeam;
constexpr u32 kCandidateRunCount = 4;
constexpr u32 kStage3Tile = 32;
constexpr u32 kRandomCases = 128;
constexpr u32 kStreamingCandidateCapacity =
  gpu_search::kPersistentMaxMergeCandidates;
constexpr u32 kStreamingBoundaryCount = 8;
constexpr u32 kStreamingExpansionModes = 4;
constexpr u32 kStreamingCases =
  kStreamingBoundaryCount * kStreamingExpansionModes;
constexpr u32 kStreamingCandidateCounts[kStreamingBoundaryCount]{
  0u, 1u, 511u, 512u, 513u, 1024u, 1536u, 2048u};
constexpr u32 kLeafRegressionCtas = 32;
constexpr u32 kLeafRegressionIterations = 16;
constexpr u32 kLeafRegressionInputSize =
  kApproximateSortThreadsCompact *
  kApproximateSortItemsCompactFinal256;
constexpr u32 kLeafRegressionOutputSize = kBeamCapacity;
constexpr u32 kLeafRegressionCount =
  kLeafRegressionCtas * kLeafRegressionIterations;
constexpr u32 kLeafCanaryCta = 13;
constexpr u32 kLeafCanaryIteration = 7;
constexpr u32 kLeafCanaryOrdinal = 428;
constexpr u32 kLeafCanaryDistanceBits = 0x474c2743u;
constexpr u64 kLeafCanaryHandle = 0xfedcba9876543210ull;
constexpr u64 kObservedCorruptKeyHandle =
  (u64{kLeafCanaryOrdinal} << 32) | kLeafCanaryDistanceBits;

struct CaseResult {
  u32 mismatch{};
  u32 expected_prefix_count{};
  u32 actual_prefix_count{};
  u32 published_ranks{};
};

struct StreamingCaseResult {
  u32 mismatch{};
  u32 candidate_count{};
  u32 issue_count{};
  u32 reusable_prefix{};
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

__global__ void stable_candidate_leaf_regression_kernel(
    const u64* candidate_handles, const f32* candidate_distances,
    const u32* candidate_counts, u64* output_handles,
    u8* output_flags, f32* output_distances) {
  __shared__
    ApproximateBlockSortCompactFinal256::TempStorage radix_storage;

  const u32 cta = blockIdx.x;
  for (u32 iteration = 0;
       iteration < kLeafRegressionIterations; ++iteration) {
    const u32 leaf = cta * kLeafRegressionIterations + iteration;
    const u32 input_offset = leaf * kLeafRegressionInputSize;
    const u32 output_offset = leaf * kLeafRegressionOutputSize;
    stable_sort_candidate_run<ApproximateBlockSortCompactFinal256,
                              kApproximateSortItemsCompactFinal256>(
      radix_storage, candidate_handles, candidate_distances,
      input_offset + candidate_counts[leaf], input_offset,
      output_handles, output_flags, output_distances,
      output_offset, kLeafRegressionOutputSize, nullptr, nullptr);
  }
}

__global__ void fused_prefix_equivalence_kernel(
    const u64* input_beam_handles, const f32* input_beam_distances,
    const u8* input_beam_expanded, const u32* input_beam_counts,
    const u64* scratch_handles, const u8* scratch_flags,
    const f32* scratch_distances, const u32* issue_capacities,
    CaseResult* results) {
  __shared__ u64 reference_beam_handles[kBeamCapacity];
  __shared__ u32 reference_beam_ids[kBeamCapacity];
  __shared__ f32 reference_beam_distances[kBeamCapacity];
  __shared__ u8 reference_beam_expanded[kBeamCapacity];
  __shared__ u64 phased_beam_handles[kBeamCapacity];
  __shared__ u32 phased_beam_ids[kBeamCapacity];
  __shared__ f32 phased_beam_distances[kBeamCapacity];
  __shared__ u8 phased_beam_expanded[kBeamCapacity];
  __shared__ CandidateWorkspaceArrays reference_workspace;
  __shared__ CandidateWorkspaceArrays phased_workspace;
  __shared__ u64 expected_prefix_handles[kStage3Tile];
  __shared__ u16 expected_prefix_ranks[kStage3Tile];
  __shared__ u64 actual_prefix_handles[kStage3Tile];
  __shared__ u16 actual_prefix_ranks[kStage3Tile];
  __shared__ u32 reference_beam_count;
  __shared__ u32 phased_beam_count;
  __shared__ u32 expected_prefix_count;
  __shared__ u32 actual_prefix_count;
  __shared__ u32 published_ranks;
  __shared__ u32 mismatch;
  __shared__ StableMergePreparedState phased_state;

  const u32 case_index = blockIdx.x;
  const u32 beam_offset = case_index * kBeamCapacity;
  const u32 scratch_offset =
    case_index * kCandidateRunCount * kBeamCapacity;
  const u64* case_scratch_handles = scratch_handles + scratch_offset;
  const u8* case_scratch_flags = scratch_flags + scratch_offset;
  const f32* case_scratch_distances = scratch_distances + scratch_offset;
  const u32 origin_count = input_beam_counts[case_index];
  const u32 issue_capacity = issue_capacities[case_index];
  const u32 candidate_run_count =
    case_index < 4u ? case_index + 1u : kCandidateRunCount;

  for (u32 rank = threadIdx.x; rank < kBeamCapacity;
       rank += blockDim.x) {
    const u64 handle = input_beam_handles[beam_offset + rank];
    const f32 distance = input_beam_distances[beam_offset + rank];
    const u8 expanded = input_beam_expanded[beam_offset + rank];
    reference_beam_handles[rank] = handle;
    reference_beam_ids[rank] = rank;
    reference_beam_distances[rank] = distance;
    reference_beam_expanded[rank] = expanded;
    phased_beam_handles[rank] = handle;
    phased_beam_ids[rank] = rank;
    phased_beam_distances[rank] = distance;
    phased_beam_expanded[rank] = expanded;
  }
  if (threadIdx.x == 0) {
    reference_beam_count = origin_count;
    phased_beam_count = origin_count;
    expected_prefix_count = 0;
    actual_prefix_count = 0;
    published_ranks = 0;
    mismatch = 0;
    phased_state = {};
  }
  __syncthreads();

  // This is the current production exact certificate. The new fused begin
  // must expose the identical handles and authoritative Beam ranks.
  preview_serial_stable_unexpanded_frontier(
    reference_beam_handles, reference_beam_distances,
    reference_beam_expanded, reference_beam_count, kBeamCapacity,
    case_scratch_handles, case_scratch_flags, case_scratch_distances,
    candidate_run_count, issue_capacity,
    expected_prefix_handles, expected_prefix_ranks,
    expected_prefix_count);

  // Keep the reference independent from the production final-fold dispatcher.
  // In particular, an empty-run fast path in
  // materialize_fused_stable_candidate_runs must be compared with the original
  // per-rank co-rank algorithm rather than with another invocation of itself.
  extend_fused_stable_candidate_tree(
    reference_beam_distances, kBeamCapacity, origin_count,
    reference_beam_handles, reference_beam_expanded,
    case_scratch_handles, case_scratch_flags, case_scratch_distances,
    candidate_run_count, reference_workspace, 0, kBeamCapacity);
  const u32 reference_run3_count =
    candidate_run_count > 3u ? kBeamCapacity : 0u;
  for (u32 rank = threadIdx.x; rank < kBeamCapacity;
       rank += blockDim.x) {
    const u32 prefix_index = stable_merge_a_corank(
      rank, reference_workspace.distances + 2u * kBeamCapacity,
      kBeamCapacity,
      case_scratch_distances + 3u * kBeamCapacity,
      reference_run3_count);
    const u32 run3_index = rank - prefix_index;
    const bool take_prefix =
      prefix_index < kBeamCapacity &&
      (run3_index >= reference_run3_count ||
       !(case_scratch_distances[
           3u * kBeamCapacity + run3_index] <
         reference_workspace.distances[
           2u * kBeamCapacity + prefix_index]));
    if (take_prefix) {
      const u32 source = 2u * kBeamCapacity + prefix_index;
      reference_beam_handles[rank] =
        reference_workspace.handles[source];
      reference_beam_distances[rank] =
        reference_workspace.distances[source];
      reference_beam_expanded[rank] =
        reference_workspace.expanded[source];
    } else {
      const u32 source = 3u * kBeamCapacity + run3_index;
      reference_beam_handles[rank] = case_scratch_handles[source];
      reference_beam_distances[rank] = case_scratch_distances[source];
      reference_beam_expanded[rank] = case_scratch_flags[source];
    }
    reference_beam_ids[rank] = UINT32_MAX;
  }
  __syncthreads();
  finalize_fused_stable_candidate_runs(
    reference_beam_handles, reference_beam_distances,
    reference_beam_count, kBeamCapacity);

  begin_fused_stable_frontier_materialization(
    phased_beam_handles, phased_beam_ids,
    phased_beam_distances, phased_beam_expanded,
    kBeamCapacity, origin_count,
    phased_beam_handles, phased_beam_expanded,
    case_scratch_handles, case_scratch_flags, case_scratch_distances,
    candidate_run_count, phased_workspace, issue_capacity,
    actual_prefix_handles, actual_prefix_ranks, actual_prefix_count,
    phased_state);

  if (threadIdx.x == 0) {
    published_ranks = phased_state.materialized_prefix;
    if (expected_prefix_count != actual_prefix_count) {
      mismatch |= 1u;
    }
    const u32 compared =
      min(expected_prefix_count, actual_prefix_count);
    for (u32 index = 0; index < compared; ++index) {
      if (expected_prefix_handles[index] != actual_prefix_handles[index] ||
          expected_prefix_ranks[index] != actual_prefix_ranks[index]) {
        mismatch |= 1u;
        break;
      }
    }
    if (published_ranks > kBeamCapacity ||
        (published_ranks != kBeamCapacity &&
         published_ranks % kStage3Tile != 0)) {
      mismatch |= 8u;
    }
  }
  __syncthreads();

  for (u32 rank = threadIdx.x;
       rank < published_ranks;
       rank += blockDim.x) {
    if (reference_beam_handles[rank] != phased_beam_handles[rank] ||
        __float_as_uint(reference_beam_distances[rank]) !=
          __float_as_uint(phased_beam_distances[rank]) ||
        reference_beam_expanded[rank] != phased_beam_expanded[rank] ||
        reference_beam_ids[rank] != phased_beam_ids[rank]) {
      atomicOr(&mismatch, 2u);
    }
  }
  __syncthreads();

  finish_fused_stable_frontier_materialization(
    phased_beam_handles, phased_beam_ids,
    phased_beam_distances, phased_beam_expanded,
    phased_beam_count, kBeamCapacity,
    case_scratch_handles, case_scratch_flags, case_scratch_distances,
    candidate_run_count, phased_workspace, phased_state);

  for (u32 rank = threadIdx.x; rank < kBeamCapacity;
       rank += blockDim.x) {
    if (reference_beam_handles[rank] != phased_beam_handles[rank] ||
        __float_as_uint(reference_beam_distances[rank]) !=
          __float_as_uint(phased_beam_distances[rank]) ||
        reference_beam_expanded[rank] != phased_beam_expanded[rank] ||
        reference_beam_ids[rank] != phased_beam_ids[rank]) {
      atomicOr(&mismatch, 4u);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reference_beam_count != phased_beam_count) mismatch |= 4u;
    actual_prefix_count = 0;
    published_ranks = 0;
    phased_beam_count = origin_count;
    phased_state = {};
    phased_state.original_count = origin_count;
    phased_state.candidate_run_count = candidate_run_count;
    phased_state.compact = 1;
    phased_state.prepared = 1;
  }
  for (u32 rank = threadIdx.x; rank < kBeamCapacity;
       rank += blockDim.x) {
    phased_beam_handles[rank] =
      input_beam_handles[beam_offset + rank];
    phased_beam_ids[rank] = rank;
    phased_beam_distances[rank] =
      input_beam_distances[beam_offset + rank];
    phased_beam_expanded[rank] =
      input_beam_expanded[beam_offset + rank];
  }
  __syncthreads();

  // Build a read-only, dependency-closed certificate whose internal/final
  // merge prefix is retained for finish.  Validate independently that the
  // certificate matches the serial oracle, that no authoritative Beam byte is
  // published here, and that finish reuses the prefix without changing the
  // complete Stable-Run result.
  prepare_reusable_fused_frontier_certificate(
    phased_beam_handles, phased_beam_distances, phased_beam_expanded,
    kBeamCapacity, origin_count,
    case_scratch_handles, case_scratch_flags, case_scratch_distances,
    candidate_run_count, phased_workspace, issue_capacity,
    actual_prefix_handles, actual_prefix_ranks, actual_prefix_count,
    phased_state);

  if (threadIdx.x == 0) {
    published_ranks = phased_state.materialized_prefix;
    if (expected_prefix_count != actual_prefix_count) {
      mismatch |= 16u;
    }
    const u32 compared =
      min(expected_prefix_count, actual_prefix_count);
    for (u32 index = 0; index < compared; ++index) {
      if (expected_prefix_handles[index] != actual_prefix_handles[index] ||
          expected_prefix_ranks[index] != actual_prefix_ranks[index]) {
        mismatch |= 16u;
        break;
      }
    }
    if (published_ranks > kBeamCapacity ||
        published_ranks < issue_capacity ||
        phased_state.deferred_prefix != 1u ||
        phased_state.fused_tree_prefix != published_ranks) {
      mismatch |= 16u;
    }
  }
  for (u32 rank = threadIdx.x; rank < kBeamCapacity;
       rank += blockDim.x) {
    if (input_beam_handles[beam_offset + rank] !=
          phased_beam_handles[rank] ||
        __float_as_uint(input_beam_distances[beam_offset + rank]) !=
          __float_as_uint(phased_beam_distances[rank]) ||
        input_beam_expanded[beam_offset + rank] !=
          phased_beam_expanded[rank] ||
        phased_beam_ids[rank] != rank) {
      atomicOr(&mismatch, 32u);
    }
  }
  __syncthreads();

  finish_fused_stable_frontier_materialization(
    phased_beam_handles, phased_beam_ids,
    phased_beam_distances, phased_beam_expanded,
    phased_beam_count, kBeamCapacity,
    case_scratch_handles, case_scratch_flags, case_scratch_distances,
    candidate_run_count, phased_workspace, phased_state);

  for (u32 rank = threadIdx.x; rank < kBeamCapacity;
       rank += blockDim.x) {
    if (reference_beam_handles[rank] != phased_beam_handles[rank] ||
        __float_as_uint(reference_beam_distances[rank]) !=
          __float_as_uint(phased_beam_distances[rank]) ||
        reference_beam_expanded[rank] != phased_beam_expanded[rank] ||
        reference_beam_ids[rank] != phased_beam_ids[rank]) {
      atomicOr(&mismatch, 64u);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reference_beam_count != phased_beam_count) mismatch |= 64u;
    actual_prefix_count = 0;
    published_ranks = 0;
    phased_beam_count = origin_count;
    phased_state = {};
    // Production obtains these fields from
    // prepare_approximate_stable_runs() before asking for the deferred
    // certificate.  The leaves supplied to this focused kernel are already
    // sorted, so initialize the same post-prepare contract explicitly.
    phased_state.original_count = origin_count;
    phased_state.candidate_run_count = candidate_run_count;
    phased_state.compact = 1;
    phased_state.prepared = 1;
  }
  for (u32 rank = threadIdx.x; rank < kBeamCapacity;
       rank += blockDim.x) {
    phased_beam_handles[rank] =
      input_beam_handles[beam_offset + rank];
    phased_beam_ids[rank] = rank;
    phased_beam_distances[rank] =
      input_beam_distances[beam_offset + rank];
    phased_beam_expanded[rank] =
      input_beam_expanded[beam_offset + rank];
  }
  __syncthreads();

  // Deferred certificate: produce the same exact Issue Frontier while
  // retaining no authoritative tree prefix.  This is the intended production
  // overlap path: Beam stays immutable until finish rebuilds [0, K).
  prepare_deferred_fused_frontier_certificate(
    phased_beam_handles, phased_beam_distances, phased_beam_expanded,
    kBeamCapacity, origin_count,
    case_scratch_handles, case_scratch_flags, case_scratch_distances,
    candidate_run_count, phased_workspace, issue_capacity,
    actual_prefix_handles, actual_prefix_ranks, actual_prefix_count,
    phased_state);

  if (threadIdx.x == 0) {
    if (expected_prefix_count != actual_prefix_count) {
      mismatch |= 128u;
    }
    const u32 compared =
      min(expected_prefix_count, actual_prefix_count);
    for (u32 index = 0; index < compared; ++index) {
      if (expected_prefix_handles[index] != actual_prefix_handles[index] ||
          expected_prefix_ranks[index] != actual_prefix_ranks[index]) {
        mismatch |= 128u;
        break;
      }
    }
    if (phased_state.fused_tree_prepared != 1u ||
        phased_state.fused_tree_prefix != 0u ||
        phased_state.materialized_prefix != 0u ||
        phased_state.deferred_prefix != 1u) {
      mismatch |= 128u;
    }
  }
  for (u32 rank = threadIdx.x; rank < kBeamCapacity;
       rank += blockDim.x) {
    if (input_beam_handles[beam_offset + rank] !=
          phased_beam_handles[rank] ||
        __float_as_uint(input_beam_distances[beam_offset + rank]) !=
          __float_as_uint(phased_beam_distances[rank]) ||
        input_beam_expanded[beam_offset + rank] !=
          phased_beam_expanded[rank] ||
        phased_beam_ids[rank] != rank) {
      atomicOr(&mismatch, 256u);
    }
  }
  __syncthreads();

  finish_fused_stable_frontier_materialization(
    phased_beam_handles, phased_beam_ids,
    phased_beam_distances, phased_beam_expanded,
    phased_beam_count, kBeamCapacity,
    case_scratch_handles, case_scratch_flags, case_scratch_distances,
    candidate_run_count, phased_workspace, phased_state);

  for (u32 rank = threadIdx.x; rank < kBeamCapacity;
       rank += blockDim.x) {
    if (reference_beam_handles[rank] != phased_beam_handles[rank] ||
        __float_as_uint(reference_beam_distances[rank]) !=
          __float_as_uint(phased_beam_distances[rank]) ||
        reference_beam_expanded[rank] != phased_beam_expanded[rank] ||
        reference_beam_ids[rank] != phased_beam_ids[rank]) {
      atomicOr(&mismatch, 512u);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reference_beam_count != phased_beam_count ||
        phased_state.fused_tree_prepared != 0u ||
        phased_state.fused_tree_prefix != 0u ||
        phased_state.materialized_prefix != 0u ||
        phased_state.deferred_prefix != 0u) {
      mismatch |= 512u;
    }
    results[case_index] = CaseResult{
      mismatch, expected_prefix_count, actual_prefix_count,
      published_ranks};
  }
}

__device__ __forceinline__ void verify_streaming_beam_unchanged(
    const u64* input_beam_handles, const f32* input_beam_distances,
    const u8* input_beam_expanded, u32 beam_offset,
    const u64* beam_handles, const u32* beam_ids,
    const f32* beam_distances, const u8* beam_expanded,
    u32* mismatch, u32 mismatch_bit) {
  for (u32 rank = threadIdx.x; rank < kBeamCapacity;
       rank += blockDim.x) {
    if (input_beam_handles[beam_offset + rank] != beam_handles[rank] ||
        __float_as_uint(input_beam_distances[beam_offset + rank]) !=
          __float_as_uint(beam_distances[rank]) ||
        input_beam_expanded[beam_offset + rank] != beam_expanded[rank] ||
        beam_ids[rank] != rank) {
      atomicOr(mismatch, mismatch_bit);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void extend_and_verify_streaming_prefix(
    u64* candidate_handles, f32* candidate_distances,
    u32 visible_candidate_count, u32 beam_capacity,
    u64* scratch_handles, u8* scratch_flags, f32* scratch_distances,
    CandidateWorkspace& workspace, StableMergePreparedState& state,
    const u64* input_beam_handles, const f32* input_beam_distances,
    const u8* input_beam_expanded, u32 beam_offset,
    const u64* beam_handles, const u32* beam_ids,
    const f32* beam_distances, const u8* beam_expanded,
    u32* mismatch, bool seal_partial = false) {
  extend_streaming_compact_stable_runs(
    candidate_handles, candidate_distances, visible_candidate_count,
    beam_capacity, scratch_handles, scratch_flags, scratch_distances,
    workspace, state, false, nullptr, seal_partial);
  if (threadIdx.x == 0) {
    if (state.streaming_candidate_offset > visible_candidate_count ||
        state.prepared != 0 || state.compact == 0) {
      atomicOr(mismatch, 1u);
    }
  }
  __syncthreads();
  verify_streaming_beam_unchanged(
    input_beam_handles, input_beam_distances, input_beam_expanded,
    beam_offset, beam_handles, beam_ids, beam_distances, beam_expanded,
    mismatch, 2u);
}

__global__ void cossf_equivalence_kernel(
    const u64* input_beam_handles, const f32* input_beam_distances,
    const u8* input_beam_expanded, const u32* input_beam_counts,
    u64* candidate_handles, f32* candidate_distances,
    const u32* candidate_counts, const u32* issue_capacities,
    StreamingCaseResult* results) {
  __shared__ u64 legacy_beam_handles[kBeamCapacity];
  __shared__ u32 legacy_beam_ids[kBeamCapacity];
  __shared__ f32 legacy_beam_distances[kBeamCapacity];
  __shared__ u8 legacy_beam_expanded[kBeamCapacity];
  __shared__ u64 streaming_beam_handles[kBeamCapacity];
  __shared__ u32 streaming_beam_ids[kBeamCapacity];
  __shared__ f32 streaming_beam_distances[kBeamCapacity];
  __shared__ u8 streaming_beam_expanded[kBeamCapacity];
  __shared__ u64 scratch_handles[
    kCandidateRunCount * kBeamCapacity];
  __shared__ u8 scratch_flags[
    kCandidateRunCount * kBeamCapacity];
  __shared__ f32 scratch_distances[
    kCandidateRunCount * kBeamCapacity];
  __shared__ u64 issue_handles[gpu_search::kPersistentFrontierRobCapacity];
  __shared__ u16 issue_ranks[gpu_search::kPersistentFrontierRobCapacity];
  __shared__ u32 selected_beam_ranks[4];
  __shared__ CandidateWorkspace workspace;
  __shared__ StableMergePreparedState state;
  __shared__ u32 legacy_beam_count;
  __shared__ u32 streaming_beam_count;
  __shared__ u32 selected_count;
  __shared__ u32 issue_count;
  __shared__ u32 reusable_prefix;
  __shared__ u32 mismatch;

  const u32 case_index = blockIdx.x;
  const u32 beam_offset = case_index * kBeamCapacity;
  const u32 candidate_offset =
    case_index * kStreamingCandidateCapacity;
  u64* case_candidate_handles = candidate_handles + candidate_offset;
  f32* case_candidate_distances =
    candidate_distances + candidate_offset;
  const u32 candidate_count = candidate_counts[case_index];
  const u32 issue_capacity = issue_capacities[case_index];

  for (u32 rank = threadIdx.x; rank < kBeamCapacity;
       rank += blockDim.x) {
    const u64 handle = input_beam_handles[beam_offset + rank];
    const f32 distance = input_beam_distances[beam_offset + rank];
    const u8 expanded = input_beam_expanded[beam_offset + rank];
    legacy_beam_handles[rank] = handle;
    legacy_beam_ids[rank] = rank;
    legacy_beam_distances[rank] = distance;
    legacy_beam_expanded[rank] = expanded;
    streaming_beam_handles[rank] = handle;
    streaming_beam_ids[rank] = rank;
    streaming_beam_distances[rank] = distance;
    streaming_beam_expanded[rank] = expanded;
  }
  if (threadIdx.x == 0) {
    legacy_beam_count = input_beam_counts[case_index];
    streaming_beam_count = input_beam_counts[case_index];
    selected_count = min(input_beam_counts[case_index], 4u);
    for (u32 selected = 0; selected < selected_count; ++selected) {
      selected_beam_ranks[selected] = selected * 3u;
      legacy_beam_expanded[selected_beam_ranks[selected]] = 1;
    }
    issue_count = 0;
    reusable_prefix = 0;
    mismatch = 0;
    state = {};
  }
  __syncthreads();

  // Independent oracle: the ordinary one-shot Stable-Run prepare followed by
  // the original two-level finish.  Its workspace and sorted leaves are
  // deliberately reused only after the complete reference Beam is published.
  prepare_approximate_stable_runs(
    case_candidate_handles, case_candidate_distances, candidate_count,
    legacy_beam_handles, legacy_beam_ids, legacy_beam_distances,
    legacy_beam_expanded, legacy_beam_count, kBeamCapacity,
    scratch_handles, scratch_flags, scratch_distances,
    workspace, state, nullptr, false);
  finish_approximate_stable_runs(
    legacy_beam_handles, legacy_beam_ids, legacy_beam_distances,
    legacy_beam_expanded, legacy_beam_count, kBeamCapacity,
    scratch_handles, scratch_flags, scratch_distances,
    workspace, state, nullptr, false);

  // Production COSSF path. Repeated calls deliberately include duplicate and
  // partial-leaf observations. Only immutable input intervals may be sealed,
  // and no call may publish an authoritative Beam byte.
  begin_streaming_compact_stable_runs(
    streaming_beam_handles, streaming_beam_distances,
    streaming_beam_expanded, streaming_beam_count,
    selected_beam_ranks, selected_count,
    kBeamCapacity, workspace, state);
  verify_streaming_beam_unchanged(
    input_beam_handles, input_beam_distances, input_beam_expanded,
    beam_offset, streaming_beam_handles, streaming_beam_ids,
    streaming_beam_distances, streaming_beam_expanded, &mismatch, 2u);

  extend_and_verify_streaming_prefix(
    case_candidate_handles, case_candidate_distances,
    min(candidate_count, 1u), kBeamCapacity,
    scratch_handles, scratch_flags, scratch_distances, workspace, state,
    input_beam_handles, input_beam_distances, input_beam_expanded,
    beam_offset, streaming_beam_handles, streaming_beam_ids,
    streaming_beam_distances, streaming_beam_expanded, &mismatch);
  // Exercise the production COSSF property that an arbitrary immutable
  // microbatch boundary may be folded once without changing the stable total
  // order. Subsequent partial requests must not fragment the same round.
  extend_and_verify_streaming_prefix(
    case_candidate_handles, case_candidate_distances,
    min(candidate_count, 300u), kBeamCapacity,
    scratch_handles, scratch_flags, scratch_distances, workspace, state,
    input_beam_handles, input_beam_distances, input_beam_expanded,
    beam_offset, streaming_beam_handles, streaming_beam_ids,
    streaming_beam_distances, streaming_beam_expanded, &mismatch, true);
  extend_and_verify_streaming_prefix(
    case_candidate_handles, case_candidate_distances,
    min(candidate_count, 511u), kBeamCapacity,
    scratch_handles, scratch_flags, scratch_distances, workspace, state,
    input_beam_handles, input_beam_distances, input_beam_expanded,
    beam_offset, streaming_beam_handles, streaming_beam_ids,
    streaming_beam_distances, streaming_beam_expanded, &mismatch);
  extend_and_verify_streaming_prefix(
    case_candidate_handles, case_candidate_distances,
    min(candidate_count, 512u), kBeamCapacity,
    scratch_handles, scratch_flags, scratch_distances, workspace, state,
    input_beam_handles, input_beam_distances, input_beam_expanded,
    beam_offset, streaming_beam_handles, streaming_beam_ids,
    streaming_beam_distances, streaming_beam_expanded, &mismatch);
  // Exercise the idempotent no-new-leaf path explicitly.
  extend_and_verify_streaming_prefix(
    case_candidate_handles, case_candidate_distances,
    min(candidate_count, 512u), kBeamCapacity,
    scratch_handles, scratch_flags, scratch_distances, workspace, state,
    input_beam_handles, input_beam_distances, input_beam_expanded,
    beam_offset, streaming_beam_handles, streaming_beam_ids,
    streaming_beam_distances, streaming_beam_expanded, &mismatch);
  extend_and_verify_streaming_prefix(
    case_candidate_handles, case_candidate_distances,
    min(candidate_count, 513u), kBeamCapacity,
    scratch_handles, scratch_flags, scratch_distances, workspace, state,
    input_beam_handles, input_beam_distances, input_beam_expanded,
    beam_offset, streaming_beam_handles, streaming_beam_ids,
    streaming_beam_distances, streaming_beam_expanded, &mismatch);
  extend_and_verify_streaming_prefix(
    case_candidate_handles, case_candidate_distances,
    min(candidate_count, 1024u), kBeamCapacity,
    scratch_handles, scratch_flags, scratch_distances, workspace, state,
    input_beam_handles, input_beam_distances, input_beam_expanded,
    beam_offset, streaming_beam_handles, streaming_beam_ids,
    streaming_beam_distances, streaming_beam_expanded, &mismatch);
  extend_and_verify_streaming_prefix(
    case_candidate_handles, case_candidate_distances,
    min(candidate_count, 1536u), kBeamCapacity,
    scratch_handles, scratch_flags, scratch_distances, workspace, state,
    input_beam_handles, input_beam_distances, input_beam_expanded,
    beam_offset, streaming_beam_handles, streaming_beam_ids,
    streaming_beam_distances, streaming_beam_expanded, &mismatch);
  extend_and_verify_streaming_prefix(
    case_candidate_handles, case_candidate_distances,
    candidate_count, kBeamCapacity,
    scratch_handles, scratch_flags, scratch_distances, workspace, state,
    input_beam_handles, input_beam_distances, input_beam_expanded,
    beam_offset, streaming_beam_handles, streaming_beam_ids,
    streaming_beam_distances, streaming_beam_expanded, &mismatch);

  extend_streaming_compact_stable_runs(
    case_candidate_handles, case_candidate_distances, candidate_count,
    kBeamCapacity, scratch_handles, scratch_flags, scratch_distances,
    workspace, state, true);
  const u32 early_partial = min(candidate_count, 300u);
  const u32 expected_streaming_runs = early_partial == 0 ? 0u :
    1u + (candidate_count - early_partial + 511u) / 512u;
  if (threadIdx.x == 0 &&
      (state.candidate_run_count != expected_streaming_runs ||
       state.prepared == 0 || state.compact == 0 ||
       state.original_count != streaming_beam_count)) {
    atomicOr(&mismatch, 1u);
  }
  __syncthreads();
  verify_streaming_beam_unchanged(
    input_beam_handles, input_beam_distances, input_beam_expanded,
    beam_offset, streaming_beam_handles, streaming_beam_ids,
    streaming_beam_distances, streaming_beam_expanded, &mismatch, 2u);

  prepare_streaming_stable_fold_frontier_certificate(
    kBeamCapacity, issue_capacity, workspace.arrays,
    issue_handles, issue_ranks, issue_count, state);

  if (threadIdx.x == 0) {
    if (state.fused_tree_prepared == 0 ||
        state.materialized_prefix != kBeamCapacity ||
        state.fused_tree_prefix != kBeamCapacity ||
        issue_count > issue_capacity) {
      mismatch |= 4u;
    }
    reusable_prefix = state.materialized_prefix;
    u32 previous_rank = 0;
    for (u32 issue = 0; issue < issue_count; ++issue) {
      const u32 rank = static_cast<u32>(issue_ranks[issue]);
      const u32 source =
        state.streaming_accumulator_segment * kBeamCapacity + rank;
      if (rank >= kBeamCapacity ||
          (issue != 0 && rank <= previous_rank) ||
          issue_handles[issue] != workspace.arrays.handles[source] ||
          workspace.arrays.expanded[source] != 0 ||
          !stable_run_item_valid(
            workspace.arrays.handles[source],
            workspace.arrays.distances[source])) {
        mismatch |= 4u;
        break;
      }
      previous_rank = rank;
    }
  }
  __syncthreads();
  // The certificate is extracted from the private exact accumulator. Beam
  // must still be the byte-identical old authoritative state at this point.
  verify_streaming_beam_unchanged(
    input_beam_handles, input_beam_distances, input_beam_expanded,
    beam_offset, streaming_beam_handles, streaming_beam_ids,
    streaming_beam_distances, streaming_beam_expanded, &mismatch, 8u);

  finish_streaming_stable_fold(
    streaming_beam_handles, streaming_beam_ids,
    streaming_beam_distances, streaming_beam_expanded,
    streaming_beam_count, kBeamCapacity,
    workspace.arrays, state);

  for (u32 rank = threadIdx.x; rank < kBeamCapacity;
       rank += blockDim.x) {
    if (legacy_beam_handles[rank] != streaming_beam_handles[rank] ||
        __float_as_uint(legacy_beam_distances[rank]) !=
          __float_as_uint(streaming_beam_distances[rank]) ||
        legacy_beam_expanded[rank] != streaming_beam_expanded[rank] ||
        legacy_beam_ids[rank] != streaming_beam_ids[rank]) {
      atomicOr(&mismatch, 16u);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (legacy_beam_count != streaming_beam_count) mismatch |= 16u;
    results[case_index] = StreamingCaseResult{
      mismatch, candidate_count, issue_count, reusable_prefix};
  }
}

struct HostItem {
  u64 handle{};
  f32 distance{};
  u8 expanded{};
};

f32 random_tied_distance(std::mt19937& generator) {
  const u32 selector = generator() % 32u;
  if (selector == 0) return -0.0f;
  if (selector == 1) return 0.0f;
  return static_cast<f32>(generator() % 97u) * 0.125f;
}

void fill_sorted_run(
    std::mt19937& generator, u32 case_index, u32 source,
    u32 valid_count, u64* handles, f32* distances, u8* expanded,
    bool old_beam) {
  std::vector<HostItem> items;
  items.reserve(valid_count);
  for (u32 index = 0; index < valid_count; ++index) {
    const u64 handle =
      (u64{case_index + 1u} << 40) |
      (u64{source + 1u} << 32) | u64{index + 1u};
    items.push_back(HostItem{
      handle, random_tied_distance(generator),
      static_cast<u8>(old_beam && generator() % 5u == 0)});
  }
  std::stable_sort(
    items.begin(), items.end(),
    [](const HostItem& lhs, const HostItem& rhs) {
      return lhs.distance < rhs.distance;
    });
  for (u32 index = 0; index < kBeamCapacity; ++index) {
    if (index < valid_count) {
      handles[index] = items[index].handle;
      distances[index] = items[index].distance;
      expanded[index] = items[index].expanded;
    } else {
      handles[index] = kInvalidDeviceHandle;
      distances[index] = FLT_MAX;
      expanded[index] = 0;
    }
  }
}

f32 host_float_from_bits(u32 bits) {
  f32 value{};
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

u32 host_float_bits(f32 value) {
  u32 bits{};
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void run_stable_candidate_leaf_regression() {
  const size_t input_items =
    static_cast<size_t>(kLeafRegressionCount) *
    kLeafRegressionInputSize;
  const size_t output_items =
    static_cast<size_t>(kLeafRegressionCount) *
    kLeafRegressionOutputSize;
  std::vector<u64> candidate_handles(
    input_items, kInvalidDeviceHandle);
  std::vector<f32> candidate_distances(input_items, FLT_MAX);
  std::vector<u32> candidate_counts(kLeafRegressionCount);
  constexpr u32 count_pattern[kLeafRegressionIterations]{
    512u, 511u, 429u, 128u, 127u, 1u, 300u, 0u,
    64u, 257u, 430u, 2u, 384u, 509u, 129u, 512u};
  const f32 canary_distance =
    host_float_from_bits(kLeafCanaryDistanceBits);
  const u32 canary_leaf =
    kLeafCanaryCta * kLeafRegressionIterations +
    kLeafCanaryIteration;

  for (u32 leaf = 0; leaf < kLeafRegressionCount; ++leaf) {
    u32 candidate_count =
      count_pattern[leaf % kLeafRegressionIterations];
    if (leaf == canary_leaf) candidate_count = kLeafRegressionInputSize;
    candidate_counts[leaf] = candidate_count;
    const size_t input_offset =
      static_cast<size_t>(leaf) * kLeafRegressionInputSize;
    for (u32 ordinal = 0; ordinal < candidate_count; ++ordinal) {
      u64 handle =
        0xd700000000000000ull |
        (static_cast<u64>(leaf) << 16) | u64{ordinal + 1u};
      f32 distance{};
      if (leaf == canary_leaf) {
        if (ordinal < 64u) {
          distance =
            canary_distance - 1024.0f + static_cast<f32>(ordinal);
        } else if (ordinal == kLeafCanaryOrdinal) {
          handle = kLeafCanaryHandle;
          distance = canary_distance;
        } else {
          distance =
            canary_distance + 1024.0f +
            static_cast<f32>((ordinal * 17u) % 64u);
        }
      } else {
        const u32 selector =
          (ordinal * 37u + leaf * 19u) % 257u;
        if (selector == 0u) {
          handle = kInvalidDeviceHandle;
          distance = -8.0f;
        } else if (selector == 1u) {
          distance = host_float_from_bits(0x7fc00001u);
        } else if (selector == 2u) {
          distance = host_float_from_bits(0x7f800000u);
        } else if ((selector % 31u) == 0u) {
          distance = (ordinal & 1u) == 0u ? -0.0f : 0.0f;
        } else {
          // A small value domain intentionally creates many distance ties;
          // the oracle below resolves every tie by the immutable leaf
          // ordinal, exactly like the packed Stable-Run leaf key.
          distance = static_cast<f32>(selector % 23u) * 0.125f;
        }
      }
      candidate_handles[input_offset + ordinal] = handle;
      candidate_distances[input_offset + ordinal] = distance;
    }
  }

  DeviceBuffer<u64> d_candidate_handles(input_items);
  DeviceBuffer<f32> d_candidate_distances(input_items);
  DeviceBuffer<u32> d_candidate_counts(kLeafRegressionCount);
  DeviceBuffer<u64> d_output_handles(output_items);
  DeviceBuffer<u8> d_output_flags(output_items);
  DeviceBuffer<f32> d_output_distances(output_items);
  upload(d_candidate_handles, candidate_handles);
  upload(d_candidate_distances, candidate_distances);
  upload(d_candidate_counts, candidate_counts);

  stable_candidate_leaf_regression_kernel
    <<<kLeafRegressionCtas, kApproximateSortThreadsCompact>>>(
      d_candidate_handles.get(), d_candidate_distances.get(),
      d_candidate_counts.get(), d_output_handles.get(),
      d_output_flags.get(), d_output_distances.get());
  check_cuda(
    cudaGetLastError(),
    "stable_candidate_leaf_regression_kernel launch");
  check_cuda(
    cudaDeviceSynchronize(),
    "stable_candidate_leaf_regression_kernel synchronize");

  const auto output_handles =
    download(d_output_handles, output_items);
  const auto output_flags =
    download(d_output_flags, output_items);
  const auto output_distances =
    download(d_output_distances, output_items);
  struct OracleItem {
    f32 distance;
    u32 ordinal;
  };
  bool canary_seen = false;
  for (u32 leaf = 0; leaf < kLeafRegressionCount; ++leaf) {
    const size_t input_offset =
      static_cast<size_t>(leaf) * kLeafRegressionInputSize;
    const size_t output_offset =
      static_cast<size_t>(leaf) * kLeafRegressionOutputSize;
    std::vector<OracleItem> oracle(kLeafRegressionInputSize);
    for (u32 ordinal = 0;
         ordinal < kLeafRegressionInputSize; ++ordinal) {
      f32 distance = FLT_MAX;
      if (ordinal < candidate_counts[leaf]) {
        const u64 handle = candidate_handles[input_offset + ordinal];
        distance = candidate_distances[input_offset + ordinal];
        if (handle == kInvalidDeviceHandle ||
            !std::isfinite(distance)) {
          distance = FLT_MAX;
        }
      }
      oracle[ordinal] = OracleItem{distance, ordinal};
    }
    std::sort(
      oracle.begin(), oracle.end(),
      [](const OracleItem& lhs, const OracleItem& rhs) {
        if (lhs.distance < rhs.distance) return true;
        if (rhs.distance < lhs.distance) return false;
        return lhs.ordinal < rhs.ordinal;
      });

    for (u32 rank = 0; rank < kLeafRegressionOutputSize; ++rank) {
      const OracleItem item = oracle[rank];
      const size_t source = input_offset + item.ordinal;
      const bool source_valid =
        item.ordinal < candidate_counts[leaf] &&
        std::isfinite(item.distance) && item.distance != FLT_MAX;
      const u64 expected_handle =
        source_valid ? candidate_handles[source] :
                       kInvalidDeviceHandle;
      const f32 expected_distance = item.distance;
      const u64 actual_handle = output_handles[output_offset + rank];
      const f32 actual_distance =
        output_distances[output_offset + rank];

      bool belongs_to_leaf =
        actual_handle == kInvalidDeviceHandle;
      for (u32 ordinal = 0;
           !belongs_to_leaf && ordinal < candidate_counts[leaf];
           ++ordinal) {
        belongs_to_leaf =
          candidate_handles[input_offset + ordinal] == actual_handle;
      }
      if (!belongs_to_leaf) {
        throw std::runtime_error(
          "stable candidate leaf emitted a foreign handle: leaf=" +
          std::to_string(leaf) + ", rank=" + std::to_string(rank) +
          ", handle=" + std::to_string(actual_handle));
      }
      if (actual_handle == kObservedCorruptKeyHandle) {
        throw std::runtime_error(
          "stable candidate leaf reproduced key-as-handle corruption "
          "0x000001ac474c2743 at leaf=" + std::to_string(leaf) +
          ", rank=" + std::to_string(rank));
      }
      if (actual_handle != expected_handle ||
          host_float_bits(actual_distance) !=
            host_float_bits(expected_distance) ||
          output_flags[output_offset + rank] != 0) {
        throw std::runtime_error(
          "stable candidate leaf top-K mismatch: leaf=" +
          std::to_string(leaf) + ", rank=" + std::to_string(rank) +
          ", expected_ordinal=" + std::to_string(item.ordinal) +
          ", expected_handle=" + std::to_string(expected_handle) +
          ", actual_handle=" + std::to_string(actual_handle) +
          ", expected_distance_bits=" +
          std::to_string(host_float_bits(expected_distance)) +
          ", actual_distance_bits=" +
          std::to_string(host_float_bits(actual_distance)));
      }
      if (actual_handle == kLeafCanaryHandle) {
        if (leaf != canary_leaf || canary_seen) {
          throw std::runtime_error(
            "stable candidate leaf duplicated or cross-contaminated "
            "the unique canary handle");
        }
        canary_seen = true;
      }
    }
  }
  if (!canary_seen) {
    throw std::runtime_error(
      "stable candidate leaf lost the ordinal-428 canary handle");
  }
  std::cout
    << "PASS: stable radix leaf preserved exact top-K and handle "
       "provenance across " << kLeafRegressionCtas << " CTAs x "
    << kLeafRegressionIterations
    << " repeated shared-storage leaves; ordinal-428 distance-key "
       "canary remained a handle and 0x000001ac474c2743 never appeared\n";
}

void run_random_equivalence_cases() {
  const size_t beam_items =
    static_cast<size_t>(kRandomCases) * kBeamCapacity;
  const size_t scratch_items =
    beam_items * kCandidateRunCount;
  std::vector<u64> beam_handles(beam_items);
  std::vector<f32> beam_distances(beam_items);
  std::vector<u8> beam_expanded(beam_items);
  std::vector<u32> beam_counts(kRandomCases);
  std::vector<u64> scratch_handles(scratch_items);
  std::vector<f32> scratch_distances(scratch_items);
  std::vector<u8> scratch_flags(scratch_items);
  std::vector<u32> issue_capacities(kRandomCases);

  std::mt19937 generator(0x5eedf00du);
  constexpr u32 issue_choices[]{1u, 8u, 16u, 31u, 32u};
  for (u32 case_index = 0; case_index < kRandomCases; ++case_index) {
    const u32 old_count = case_index == 2u ? 0u :
      (case_index == 3u ? kBeamCapacity :
       48u + generator() % 81u);
    beam_counts[case_index] = old_count;
    issue_capacities[case_index] =
      issue_choices[case_index % std::size(issue_choices)];
    const size_t beam_offset =
      static_cast<size_t>(case_index) * kBeamCapacity;
    fill_sorted_run(
      generator, case_index, 0, old_count,
      beam_handles.data() + beam_offset,
      beam_distances.data() + beam_offset,
      beam_expanded.data() + beam_offset, true);
    for (u32 run = 0; run < kCandidateRunCount; ++run) {
      const u32 valid_count =
        case_index == 2u && run == 0u ? 1u :
        (case_index == 3u && run == 3u ? 0u :
         24u + generator() % 105u);
      const size_t run_offset =
        (static_cast<size_t>(case_index) * kCandidateRunCount + run) *
        kBeamCapacity;
      fill_sorted_run(
        generator, case_index, run + 1u, valid_count,
        scratch_handles.data() + run_offset,
        scratch_distances.data() + run_offset,
        scratch_flags.data() + run_offset, false);
    }
    const bool all_equal_case =
      case_index == 0u || case_index == 3u;
    const bool signed_zero_case =
      case_index == 1u || case_index == 4u;
    if (all_equal_case || signed_zero_case) {
      // Cases 3 and 4 both expose all four candidate runs: case 3 keeps run3
      // semantically empty, while case 4 makes it non-empty.  Together with
      // the reduced-run cases 0/1, all-equal and signed-zero leaves stress the
      // exact stable origin order old < run0 < run1 < run2 < run3 on both
      // sides of the empty-run dispatch. CUDA/CUB treats both signed zeros as
      // an equal key; the merge comparator must do the same bit-for-bit.
      for (u32 rank = 0; rank < old_count; ++rank) {
        beam_distances[beam_offset + rank] =
          all_equal_case ? 1.0f :
          ((rank & 1u) == 0 ? -0.0f : 0.0f);
        beam_expanded[beam_offset + rank] =
          static_cast<u8>((rank % 3u) == 0u);
      }
      for (u32 run = 0; run < kCandidateRunCount; ++run) {
        const size_t run_offset =
          (static_cast<size_t>(case_index) * kCandidateRunCount + run) *
          kBeamCapacity;
        for (u32 rank = 0; rank < kBeamCapacity; ++rank) {
          if (scratch_handles[run_offset + rank] ==
              kInvalidDeviceHandle) {
            continue;
          }
          scratch_distances[run_offset + rank] =
            all_equal_case ? 1.0f :
            ((rank & 1u) == 0 ? 0.0f : -0.0f);
        }
      }
    }
  }

  DeviceBuffer<u64> d_beam_handles(beam_items);
  DeviceBuffer<f32> d_beam_distances(beam_items);
  DeviceBuffer<u8> d_beam_expanded(beam_items);
  DeviceBuffer<u32> d_beam_counts(kRandomCases);
  DeviceBuffer<u64> d_scratch_handles(scratch_items);
  DeviceBuffer<f32> d_scratch_distances(scratch_items);
  DeviceBuffer<u8> d_scratch_flags(scratch_items);
  DeviceBuffer<u32> d_issue_capacities(kRandomCases);
  DeviceBuffer<CaseResult> d_results(kRandomCases);
  upload(d_beam_handles, beam_handles);
  upload(d_beam_distances, beam_distances);
  upload(d_beam_expanded, beam_expanded);
  upload(d_beam_counts, beam_counts);
  upload(d_scratch_handles, scratch_handles);
  upload(d_scratch_distances, scratch_distances);
  upload(d_scratch_flags, scratch_flags);
  upload(d_issue_capacities, issue_capacities);

  fused_prefix_equivalence_kernel<<<kRandomCases, 128>>>(
    d_beam_handles.get(), d_beam_distances.get(),
    d_beam_expanded.get(), d_beam_counts.get(),
    d_scratch_handles.get(), d_scratch_flags.get(),
    d_scratch_distances.get(), d_issue_capacities.get(),
    d_results.get());
  check_cuda(cudaGetLastError(), "fused_prefix_equivalence_kernel launch");
  check_cuda(
    cudaDeviceSynchronize(),
    "fused_prefix_equivalence_kernel synchronize");
  const auto results = download(d_results, kRandomCases);
  for (u32 case_index = 0; case_index < kRandomCases; ++case_index) {
    if (results[case_index].mismatch == 0) continue;
    throw std::runtime_error(
      "fused prefix case " + std::to_string(case_index) +
      " diverged: mask=" + std::to_string(results[case_index].mismatch) +
      ", expected_prefix=" +
      std::to_string(results[case_index].expected_prefix_count) +
      ", actual_prefix=" +
      std::to_string(results[case_index].actual_prefix_count) +
      ", published_ranks=" +
      std::to_string(results[case_index].published_ranks));
  }
  std::cout
    << "PASS: 128 randomized published-prefix, reusable-certificate, and "
       "deferred-certificate begin/finish cases are bitwise equivalent to "
       "full Stable-Run materialization and serial preview; both read-only "
       "certificates leave Beam byte-identical until finish\n";
}

f32 targeted_old_distance(u32 pattern, u32 rank) {
  if (pattern == 0u) return 1.0f;
  if (pattern == 1u) return (rank & 1u) == 0 ? -0.0f : 0.0f;
  if (pattern == 2u) return static_cast<f32>(rank / 8u) * 0.125f;
  return static_cast<f32>(rank / 3u) * 0.0625f;
}

f32 targeted_candidate_distance(
    u32 pattern, u32 case_index, u32 candidate_index) {
  if (pattern == 0u) return 1.0f;
  if (pattern == 1u) {
    return (candidate_index & 1u) == 0 ? 0.0f : -0.0f;
  }
  if (pattern == 2u) {
    return static_cast<f32>(
      (candidate_index * 37u + case_index * 13u) % 61u) * 0.125f;
  }
  return static_cast<f32>(
    (candidate_index * 17u + case_index * 5u) % 29u) * 0.0625f;
}

u8 targeted_expanded(u32 mode, u32 rank) {
  if (mode == 0u) return 0;
  if (mode == 1u) return static_cast<u8>((rank % 16u) == 0u);
  if (mode == 2u) return static_cast<u8>((rank & 1u) != 0);
  return 1;
}

void run_cossf_equivalence_cases() {
  const size_t beam_items =
    static_cast<size_t>(kStreamingCases) * kBeamCapacity;
  const size_t candidate_items =
    static_cast<size_t>(kStreamingCases) *
    kStreamingCandidateCapacity;
  std::vector<u64> beam_handles(beam_items);
  std::vector<f32> beam_distances(beam_items);
  std::vector<u8> beam_expanded(beam_items);
  std::vector<u32> beam_counts(kStreamingCases, kBeamCapacity);
  std::vector<u64> candidate_handles(
    candidate_items, kInvalidDeviceHandle);
  std::vector<f32> candidate_distances(candidate_items, FLT_MAX);
  std::vector<u32> candidate_counts(kStreamingCases);
  std::vector<u32> issue_capacities(kStreamingCases);
  constexpr u32 issue_choices[]{1u, 8u, 16u, 32u};

  for (u32 boundary = 0; boundary < kStreamingBoundaryCount; ++boundary) {
    for (u32 expanded_mode = 0;
         expanded_mode < kStreamingExpansionModes; ++expanded_mode) {
      const u32 case_index =
        boundary * kStreamingExpansionModes + expanded_mode;
      const u32 pattern = (boundary + expanded_mode) % 4u;
      candidate_counts[case_index] =
        kStreamingCandidateCounts[boundary];
      issue_capacities[case_index] = issue_choices[expanded_mode];
      const size_t beam_offset =
        static_cast<size_t>(case_index) * kBeamCapacity;
      for (u32 rank = 0; rank < kBeamCapacity; ++rank) {
        beam_handles[beam_offset + rank] =
          (u64{case_index + 1u} << 40) | u64{rank + 1u};
        beam_distances[beam_offset + rank] =
          targeted_old_distance(pattern, rank);
        beam_expanded[beam_offset + rank] =
          targeted_expanded(expanded_mode, rank);
      }

      const size_t candidate_offset =
        static_cast<size_t>(case_index) *
        kStreamingCandidateCapacity;
      for (u32 index = 0; index < candidate_counts[case_index]; ++index) {
        candidate_handles[candidate_offset + index] =
          (u64{case_index + 1u} << 40) |
          (u64{1} << 32) | u64{index + 1u};
        candidate_distances[candidate_offset + index] =
          targeted_candidate_distance(pattern, case_index, index);
      }
    }
  }

  DeviceBuffer<u64> d_beam_handles(beam_items);
  DeviceBuffer<f32> d_beam_distances(beam_items);
  DeviceBuffer<u8> d_beam_expanded(beam_items);
  DeviceBuffer<u32> d_beam_counts(kStreamingCases);
  DeviceBuffer<u64> d_candidate_handles(candidate_items);
  DeviceBuffer<f32> d_candidate_distances(candidate_items);
  DeviceBuffer<u32> d_candidate_counts(kStreamingCases);
  DeviceBuffer<u32> d_issue_capacities(kStreamingCases);
  DeviceBuffer<StreamingCaseResult> d_results(kStreamingCases);
  upload(d_beam_handles, beam_handles);
  upload(d_beam_distances, beam_distances);
  upload(d_beam_expanded, beam_expanded);
  upload(d_beam_counts, beam_counts);
  upload(d_candidate_handles, candidate_handles);
  upload(d_candidate_distances, candidate_distances);
  upload(d_candidate_counts, candidate_counts);
  upload(d_issue_capacities, issue_capacities);

  cossf_equivalence_kernel<<<kStreamingCases, 128>>>(
    d_beam_handles.get(), d_beam_distances.get(),
    d_beam_expanded.get(), d_beam_counts.get(),
    d_candidate_handles.get(), d_candidate_distances.get(),
    d_candidate_counts.get(), d_issue_capacities.get(),
    d_results.get());
  check_cuda(
    cudaGetLastError(), "cossf_equivalence_kernel launch");
  check_cuda(
    cudaDeviceSynchronize(),
    "cossf_equivalence_kernel synchronize");
  const auto results = download(d_results, kStreamingCases);
  for (u32 case_index = 0; case_index < kStreamingCases; ++case_index) {
    if (results[case_index].mismatch == 0) continue;
    throw std::runtime_error(
      "streaming SRFC case " + std::to_string(case_index) +
      " diverged: mask=" + std::to_string(results[case_index].mismatch) +
      ", candidate_count=" +
      std::to_string(results[case_index].candidate_count) +
      ", issue_count=" + std::to_string(results[case_index].issue_count) +
      ", reusable_prefix=" +
      std::to_string(results[case_index].reusable_prefix));
  }
  std::cout
    << "PASS: " << kStreamingCases
    << " targeted COSSF cases are bitwise equivalent "
       "to independent one-shot Stable-Run prepare/finish across candidate "
       "counts 0/1/511/512/513/1024/1536/2048, four old-expanded "
       "densities, ties, and signed zero; no pre-finish Beam byte changed\n";
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
    run_stable_candidate_leaf_regression();
    run_random_equivalence_cases();
    run_cossf_equivalence_cases();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
