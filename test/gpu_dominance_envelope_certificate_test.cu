#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
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
constexpr u32 kWarmupIterations = 16;
constexpr u32 kMeasuredIterations = 256;

enum class InputMode : u32 {
  sparse_hit,
  exact_capacity_hit,
  overflow,
  threshold_zero_ties,
  no_anchor_hit,
  no_anchor_overflow,
  invalid_cross_leaf_ties,
};

struct CaseSpec {
  u32 candidate_count{};
  u32 beam_capacity{};
  u32 beam_count{};
  u32 commit_capacity{};
  u32 issue_capacity{};
  InputMode mode{};
};

struct ExpectedEnvelope {
  u32 hit{};
  u32 envelope_size{};
};

struct CaseResult {
  u32 mismatch{};
  u32 first_mismatch{UINT32_MAX};
  u32 certificate_count{};
  u32 oracle_count{};
  u32 envelope_hit{};
  u32 envelope_size{};
};

struct CycleResult {
  u64 cycles{};
  u32 mismatch{};
  u32 certificate_count{};
  u32 oracle_count{};
  u32 envelope_hit{};
  u32 envelope_size{};
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
  T* get() const { return data_; }
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

__global__ void dominance_envelope_equivalence_kernel(
    const CaseSpec* specs,
    const u64* input_beam_handles, const f32* input_beam_distances,
    const u8* input_beam_expanded,
    u64* candidate_handles, f32* candidate_distances,
    u64* beam_handles, u32* beam_ids, f32* beam_distances,
    u8* beam_expanded,
    u64* scratch_handles, u8* scratch_flags, f32* scratch_distances,
    CaseResult* results) {
  __shared__ CandidateWorkspace workspace;
  __shared__ DominanceEnvelopeCertificateContext dominance_context;
  __shared__ StableMergePreparedState state;
  __shared__ u32 beam_count;
  __shared__ u32 certificate_count;
  __shared__ u32 oracle_count;
  __shared__ u32 mismatch;
  __shared__ u32 first_mismatch;
  __shared__ u32 envelope_size;
  __shared__ u32 envelope_hit;
  __shared__ u64 certificate_handles[kPersistentFrontierRobCapacity];
  __shared__ u16 certificate_ranks[kPersistentFrontierRobCapacity];
  __shared__ u64 oracle_handles[kPersistentFrontierRobCapacity];
  __shared__ u16 oracle_ranks[kPersistentFrontierRobCapacity];

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
    oracle_count = 0;
    mismatch = 0;
    first_mismatch = UINT32_MAX;
    state = {};
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    dominance_context = DominanceEnvelopeCertificateContext{
      .candidate_handles = case_candidates,
      .candidate_distances = case_candidate_distances,
      .beam_handles = case_beam_handles,
      .beam_distances = case_beam_distances,
      .beam_expanded = case_beam_expanded,
      .prefix_handles = case_scratch_handles,
      .prefix_distances = case_scratch_distances,
      .workspace = &workspace.arrays,
      .output_handles = certificate_handles,
      .output_ranks = certificate_ranks,
      .output_count = &certificate_count,
      .envelope_size_out = &envelope_size,
      .candidate_count = spec.candidate_count,
      .beam_count = beam_count,
      .beam_capacity = spec.beam_capacity,
      .commit_capacity = spec.commit_capacity,
      .issue_capacity = spec.issue_capacity,
    };
  }
  __syncthreads();
  const bool dominance_ready =
    prepare_dominance_envelope_exact_certificate(dominance_context);
  if (!dominance_ready) {
    prepare_partition_bounded_exact_certificate(
      case_candidates, case_candidate_distances, spec.candidate_count,
      case_beam_handles, case_beam_distances, case_beam_expanded,
      beam_count, spec.beam_capacity,
      case_scratch_handles, case_scratch_distances,
      workspace.arrays, spec.issue_capacity,
      certificate_handles, certificate_ranks, certificate_count);
  }
  if (threadIdx.x == 0) envelope_hit = dominance_ready ? 1u : 0u;
  __syncthreads();

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

  prepare_approximate_stable_runs(
    case_candidates, case_candidate_distances, spec.candidate_count,
    case_beam_handles, case_beam_ids, case_beam_distances,
    case_beam_expanded, beam_count, spec.beam_capacity,
    case_scratch_handles, case_scratch_flags, case_scratch_distances,
    workspace, state, nullptr, false);
  const u32 oracle_capacity = dominance_ready
    ? min(spec.issue_capacity, envelope_size)
    : spec.issue_capacity;
  preview_tree_stable_unexpanded_frontier(
    case_beam_handles, case_beam_distances, case_beam_expanded,
    beam_count, spec.beam_capacity,
    case_scratch_handles, case_scratch_distances,
    state.candidate_run_count, oracle_capacity,
    workspace.arrays,
    oracle_handles, oracle_ranks, oracle_count);

  if (threadIdx.x == 0 && certificate_count != oracle_count) {
    mismatch |= 2u;
  }
  __syncthreads();
  for (u32 output = threadIdx.x;
       output < kPersistentFrontierRobCapacity;
       output += blockDim.x) {
    if (output < certificate_count &&
        (output >= oracle_count ||
         certificate_handles[output] != oracle_handles[output] ||
         certificate_ranks[output] != oracle_ranks[output])) {
      atomicOr(&mismatch, 4u);
      atomicMin(&first_mismatch, output);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    results[case_index] = CaseResult{
      mismatch, first_mismatch, certificate_count, oracle_count,
      envelope_hit, envelope_size};
  }
}

std::vector<CaseSpec> make_specs() {
  return {
    {0u, 64u, 0u, 1u, 16u, InputMode::no_anchor_hit},
    {31u, 64u, 64u, 8u, 16u, InputMode::sparse_hit},
    {32u, 64u, 64u, 16u, 16u, InputMode::exact_capacity_hit},
    {33u, 64u, 64u, 8u, 16u, InputMode::overflow},
    {127u, 64u, 41u, 4u, 8u, InputMode::invalid_cross_leaf_ties},
    {512u, 64u, 64u, 8u, 16u, InputMode::threshold_zero_ties},
    {513u, 64u, 64u, 16u, 32u, InputMode::overflow},
    {777u, 64u, 64u, 16u, 32u, InputMode::exact_capacity_hit},
    {32u, 128u, 0u, 16u, 32u, InputMode::no_anchor_hit},
    {33u, 128u, 0u, 16u, 32u, InputMode::no_anchor_overflow},
    {511u, 128u, 127u, 1u, 8u, InputMode::sparse_hit},
    {512u, 128u, 128u, 4u, 8u, InputMode::threshold_zero_ties},
    {513u, 128u, 128u, 8u, 16u, InputMode::invalid_cross_leaf_ties},
    {1023u, 128u, 128u, 8u, 16u, InputMode::overflow},
    {1024u, 128u, 128u, 8u, 17u, InputMode::exact_capacity_hit},
    {1025u, 128u, 128u, 16u, 31u, InputMode::sparse_hit},
    {1535u, 128u, 128u, 16u, 32u, InputMode::overflow},
    {1536u, 128u, 128u, 8u, 32u, InputMode::exact_capacity_hit},
    {1537u, 128u, 128u, 8u, 32u, InputMode::invalid_cross_leaf_ties},
    {2047u, 128u, 91u, 8u, 16u, InputMode::threshold_zero_ties},
    {2048u, 128u, 128u, 8u, 16u, InputMode::sparse_hit},
    {2048u, 128u, 128u, 16u, 32u, InputMode::overflow},
  };
}

bool host_valid(u64 handle, f32 distance) {
  return handle != kInvalidDeviceHandle &&
    std::isfinite(distance) && distance != FLT_MAX;
}

ExpectedEnvelope fill_case(
    u32 case_index, const CaseSpec& spec,
    u64* beam_handles, f32* beam_distances, u8* beam_expanded,
    u64* candidate_handles, f32* candidate_distances) {
  const bool no_anchor =
    spec.mode == InputMode::no_anchor_hit ||
    spec.mode == InputMode::no_anchor_overflow;
  for (u32 rank = 0; rank < kBeamCapacity; ++rank) {
    if (rank >= spec.beam_count) {
      beam_handles[rank] = kInvalidDeviceHandle;
      beam_distances[rank] = FLT_MAX;
      beam_expanded[rank] = 0;
      continue;
    }
    beam_handles[rank] =
      (u64{case_index + 1u} << 48u) | u64{rank + 1u};
    if (spec.mode == InputMode::threshold_zero_ties) {
      beam_distances[rank] =
        rank + 1u < spec.issue_capacity
          ? -static_cast<f32>(spec.issue_capacity - rank) * 0.25f
          : ((rank & 1u) == 0 ? -0.0f : 0.0f);
      beam_expanded[rank] = 0;
    } else {
      beam_distances[rank] = static_cast<f32>(rank) * 0.5f;
      beam_expanded[rank] = static_cast<u8>(
        no_anchor || ((rank + case_index) % 7u) == 0u);
    }
  }

  u32 old_unexpanded = 0;
  f32 anchor = FLT_MAX;
  for (u32 rank = 0; rank < spec.beam_count; ++rank) {
    if (!host_valid(beam_handles[rank], beam_distances[rank]) ||
        beam_expanded[rank] != 0) {
      continue;
    }
    if (++old_unexpanded == spec.issue_capacity) {
      anchor = beam_distances[rank];
      break;
    }
  }

  u32 desired_matches = 0;
  switch (spec.mode) {
    case InputMode::sparse_hit:
      desired_matches = spec.issue_capacity / 2u;
      break;
    case InputMode::exact_capacity_hit:
      desired_matches = spec.issue_capacity;
      break;
    case InputMode::overflow:
      desired_matches = 33u;
      break;
    case InputMode::threshold_zero_ties:
      desired_matches = spec.issue_capacity / 2u;
      break;
    case InputMode::no_anchor_hit:
      desired_matches = spec.issue_capacity;
      break;
    case InputMode::no_anchor_overflow:
      desired_matches = 33u;
      break;
    case InputMode::invalid_cross_leaf_ties:
      desired_matches = spec.issue_capacity;
      break;
  }
  desired_matches = min(desired_matches, spec.candidate_count);

  for (u32 ordinal = 0; ordinal < kCandidateCapacity; ++ordinal) {
    candidate_handles[ordinal] = kInvalidDeviceHandle;
    candidate_distances[ordinal] = FLT_MAX;
    if (ordinal >= spec.candidate_count || no_anchor) continue;
    candidate_handles[ordinal] =
      (u64{case_index + 1u} << 32u) | u64{ordinal + 1u};
    if (spec.mode == InputMode::threshold_zero_ties) {
      candidate_distances[ordinal] =
        (ordinal & 1u) == 0 ? -0.0f : 0.0f;
    } else {
      candidate_distances[ordinal] =
        anchor + 64.0f + static_cast<f32>(ordinal) * 0.001f;
    }
  }

  // Spread the dominance set over all raw Stable-Run leaves. Equal
  // candidate distances exercise raw-ordinal stability across leaf merges.
  std::vector<u8> chosen(spec.candidate_count, 0);
  for (u32 item = 0; item < desired_matches; ++item) {
    u32 ordinal =
      spec.candidate_count == 0 ? 0u :
      (item * 521u + 7u) % spec.candidate_count;
    while (spec.candidate_count != 0 && chosen[ordinal] != 0) {
      ordinal = (ordinal + 1u) % spec.candidate_count;
    }
    if (spec.candidate_count == 0) break;
    chosen[ordinal] = 1;
    candidate_handles[ordinal] =
      (u64{case_index + 1u} << 32u) | u64{ordinal + 1u};
    if (no_anchor) {
      candidate_distances[ordinal] =
        static_cast<f32>(item / 3u) * 0.25f - 8.0f;
    } else if (spec.mode == InputMode::overflow) {
      candidate_distances[ordinal] =
        spec.beam_count == 0
          ? -1024.0f : beam_distances[0] - 1024.0f;
    } else if (spec.mode == InputMode::invalid_cross_leaf_ties) {
      candidate_distances[ordinal] = anchor - 1.0f;
    } else if (spec.mode == InputMode::threshold_zero_ties) {
      candidate_distances[ordinal] =
        -1.0f - static_cast<f32>(item / 3u) * 0.25f;
    } else {
      candidate_distances[ordinal] =
        anchor - 1.0f - static_cast<f32>(item / 3u) * 0.25f;
    }
  }

  if (spec.mode == InputMode::invalid_cross_leaf_ties) {
    for (u32 ordinal = 0; ordinal < spec.candidate_count; ++ordinal) {
      if (chosen[ordinal] != 0 || ordinal % 47u != 0) continue;
      switch ((ordinal / 47u) & 3u) {
        case 0:
          candidate_handles[ordinal] = kInvalidDeviceHandle;
          break;
        case 1:
          candidate_distances[ordinal] =
            std::numeric_limits<f32>::quiet_NaN();
          break;
        case 2:
          candidate_distances[ordinal] =
            std::numeric_limits<f32>::infinity();
          break;
        default:
          candidate_distances[ordinal] = FLT_MAX;
          break;
      }
    }
  }

  std::vector<f32> old_unexpanded_distances;
  for (u32 rank = 0; rank < spec.beam_count; ++rank) {
    if (host_valid(beam_handles[rank], beam_distances[rank]) &&
        beam_expanded[rank] == 0) {
      old_unexpanded_distances.push_back(beam_distances[rank]);
    }
  }
  u32 valid_candidates = 0;
  for (u32 ordinal = 0; ordinal < spec.candidate_count; ++ordinal) {
    valid_candidates += host_valid(
      candidate_handles[ordinal], candidate_distances[ordinal]);
  }
  bool selected = false;
  u32 selected_envelope_size = 0;
  const u32 anchor_count = min(
    static_cast<u32>(old_unexpanded_distances.size()),
    spec.issue_capacity);
  for (u32 anchor_index = 0; anchor_index < anchor_count;
       ++anchor_index) {
    u32 dominated = 0;
    for (u32 ordinal = 0; ordinal < spec.candidate_count; ++ordinal) {
      dominated +=
        host_valid(candidate_handles[ordinal],
                   candidate_distances[ordinal]) &&
        candidate_distances[ordinal] <
          old_unexpanded_distances[anchor_index];
    }
    const u32 envelope_size = anchor_index + 1u + dominated;
    if (envelope_size > kPersistentFrontierRobCapacity) break;
    if (envelope_size >= spec.commit_capacity) {
      selected = true;
      selected_envelope_size = envelope_size;
    }
  }
  if (!selected &&
      old_unexpanded_distances.size() < spec.commit_capacity) {
    const u32 complete_size =
      static_cast<u32>(old_unexpanded_distances.size()) +
      valid_candidates;
    if (complete_size >= spec.commit_capacity &&
        complete_size <= kPersistentFrontierRobCapacity) {
      selected = true;
      selected_envelope_size = complete_size;
    }
  }
  return ExpectedEnvelope{
    selected ? 1u : 0u,
    selected ? selected_envelope_size : 0u};
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
  std::vector<ExpectedEnvelope> expected(specs.size());
  for (u32 case_index = 0; case_index < specs.size(); ++case_index) {
    expected[case_index] = fill_case(
      case_index, specs[case_index],
      beam_handles.data() + case_index * kBeamCapacity,
      beam_distances.data() + case_index * kBeamCapacity,
      beam_expanded.data() + case_index * kBeamCapacity,
      candidate_handles.data() + case_index * kCandidateCapacity,
      candidate_distances.data() + case_index * kCandidateCapacity);
  }

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

  dominance_envelope_equivalence_kernel
    <<<static_cast<u32>(specs.size()), 128>>>(
      d_specs.get(),
      d_input_beam_handles.get(), d_input_beam_distances.get(),
      d_input_beam_expanded.get(),
      d_candidate_handles.get(), d_candidate_distances.get(),
      d_beam_handles.get(), d_beam_ids.get(), d_beam_distances.get(),
      d_beam_expanded.get(),
      d_scratch_handles.get(), d_scratch_flags.get(),
      d_scratch_distances.get(), d_results.get());
  check_cuda(cudaGetLastError(), "equivalence kernel launch");
  check_cuda(cudaDeviceSynchronize(), "equivalence kernel synchronize");

  const auto results = download(d_results, specs.size());
  u32 hits = 0;
  u32 fallbacks = 0;
  u32 exact_tails = 0;
  for (u32 case_index = 0; case_index < specs.size(); ++case_index) {
    const CaseResult result = results[case_index];
    const ExpectedEnvelope want = expected[case_index];
    if (result.envelope_hit != want.hit ||
        result.envelope_size != want.envelope_size) {
      throw std::runtime_error(
        "envelope metadata mismatch case=" + std::to_string(case_index) +
        " hit=" + std::to_string(result.envelope_hit) +
        "/" + std::to_string(want.hit) +
        " size=" + std::to_string(result.envelope_size) +
        "/" + std::to_string(want.envelope_size) +
        " commit=" + std::to_string(specs[case_index].commit_capacity) +
        " issue=" + std::to_string(specs[case_index].issue_capacity));
    }
    if (result.mismatch != 0) {
      throw std::runtime_error(
        "envelope certificate diverged case=" +
        std::to_string(case_index) +
        " mask=" + std::to_string(result.mismatch) +
        " first=" + std::to_string(result.first_mismatch) +
        " certificate=" + std::to_string(result.certificate_count) +
        " oracle=" + std::to_string(result.oracle_count));
    }
    hits += result.envelope_hit;
    fallbacks += result.envelope_hit == 0;
    exact_tails += result.envelope_hit != 0 &&
      result.envelope_size < specs[case_index].issue_capacity;
  }
  if (hits == 0 || fallbacks == 0 || exact_tails == 0) {
    throw std::runtime_error(
      "test matrix did not cover hit/fallback/exact-tail paths");
  }
  std::cout
    << "PASS: " << specs.size()
    << " production adaptive-anchor dominance-envelope "
       "boundary/tie/expanded/no-anchor/invalid cases match complete "
       "Stable-Run preview; hits=" << hits
    << " fallbacks=" << fallbacks
    << " exact_tails=" << exact_tails << '\n';
}

__global__ void dominance_envelope_cycle_kernel(
    u64* candidate_handles, f32* candidate_distances,
    u32 candidate_count, u32 commit_capacity, u32 issue_capacity,
    CycleResult* result) {
  __shared__ CandidateWorkspace workspace;
  __shared__ DominanceEnvelopeCertificateContext dominance_context;
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
  __shared__ u64 certificate_handles[kPersistentFrontierRobCapacity];
  __shared__ u16 certificate_ranks[kPersistentFrontierRobCapacity];
  __shared__ u64 oracle_handles[kPersistentFrontierRobCapacity];
  __shared__ u16 oracle_ranks[kPersistentFrontierRobCapacity];
  __shared__ u32 beam_count;
  __shared__ u32 certificate_count;
  __shared__ u32 oracle_count;
  __shared__ u32 envelope_size;
  __shared__ u32 envelope_hit;
  __shared__ u64 started;
  __shared__ u64 cycles;
  __shared__ u32 mismatch;

  for (u32 rank = threadIdx.x; rank < kBeamCapacity;
       rank += blockDim.x) {
    beam_handles[rank] = 0x100000000ull + rank;
    beam_ids[rank] = rank;
    beam_distances[rank] = static_cast<f32>(rank) * 0.5f;
    beam_expanded[rank] = static_cast<u8>((rank % 9u) == 0u);
  }
  if (threadIdx.x == 0) {
    beam_count = kBeamCapacity;
    cycles = 0;
    mismatch = 0;
    state = {};
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    dominance_context = DominanceEnvelopeCertificateContext{
      .candidate_handles = candidate_handles,
      .candidate_distances = candidate_distances,
      .beam_handles = beam_handles,
      .beam_distances = beam_distances,
      .beam_expanded = beam_expanded,
      .prefix_handles = scratch_handles,
      .prefix_distances = scratch_distances,
      .workspace = &workspace.arrays,
      .output_handles = certificate_handles,
      .output_ranks = certificate_ranks,
      .output_count = &certificate_count,
      .envelope_size_out = &envelope_size,
      .candidate_count = candidate_count,
      .beam_count = beam_count,
      .beam_capacity = kBeamCapacity,
      .commit_capacity = commit_capacity,
      .issue_capacity = issue_capacity,
    };
  }
  __syncthreads();
  for (u32 iteration = 0;
       iteration < kWarmupIterations + kMeasuredIterations;
       ++iteration) {
    if (threadIdx.x == 0) {
      certificate_count = 0;
      started = clock64();
    }
    __syncthreads();
    const bool dominance_ready =
      prepare_dominance_envelope_exact_certificate(dominance_context);
    if (!dominance_ready) {
      prepare_partition_bounded_exact_certificate(
        candidate_handles, candidate_distances, candidate_count,
        beam_handles, beam_distances, beam_expanded,
        beam_count, kBeamCapacity,
        scratch_handles, scratch_distances,
        workspace.arrays, issue_capacity,
        certificate_handles, certificate_ranks, certificate_count);
    }
    if (threadIdx.x == 0) {
      envelope_hit = dominance_ready ? 1u : 0u;
    }
    __syncthreads();
    if (threadIdx.x == 0 && iteration >= kWarmupIterations) {
      cycles += clock64() - started;
    }
    __syncthreads();
  }

  prepare_approximate_stable_runs(
    candidate_handles, candidate_distances, candidate_count,
    beam_handles, beam_ids, beam_distances, beam_expanded,
    beam_count, kBeamCapacity,
    scratch_handles, scratch_flags, scratch_distances,
    workspace, state, nullptr, false);
  const u32 oracle_capacity = envelope_hit != 0
    ? min(issue_capacity, envelope_size)
    : issue_capacity;
  preview_tree_stable_unexpanded_frontier(
    beam_handles, beam_distances, beam_expanded,
    beam_count, kBeamCapacity,
    scratch_handles, scratch_distances,
    state.candidate_run_count, oracle_capacity,
    workspace.arrays,
    oracle_handles, oracle_ranks, oracle_count);
  if (threadIdx.x == 0 && certificate_count != oracle_count) mismatch = 1;
  __syncthreads();
  for (u32 output = threadIdx.x; output < certificate_count;
       output += blockDim.x) {
    if (certificate_handles[output] != oracle_handles[output] ||
        certificate_ranks[output] != oracle_ranks[output]) {
      atomicOr(&mismatch, 2u);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    *result = CycleResult{
      cycles, mismatch, certificate_count, oracle_count,
      envelope_hit, envelope_size};
  }
}

void run_cycle_case(
    bool overflow, u32 commit_capacity, u32 issue_capacity,
    double cycles_per_microsecond) {
  constexpr u32 candidate_count = 1536;
  std::vector<u64> handles(kCandidateCapacity, kInvalidDeviceHandle);
  std::vector<f32> distances(kCandidateCapacity, FLT_MAX);
  const u32 low_count =
    overflow ? 33u : commit_capacity;
  for (u32 ordinal = 0; ordinal < candidate_count; ++ordinal) {
    handles[ordinal] = 0x200000000ull + ordinal;
    distances[ordinal] =
      ordinal < low_count
        ? (overflow
            ? -1024.0f
            : static_cast<f32>(ordinal / 3u) * 0.125f)
        : 1000.0f + static_cast<f32>(ordinal) * 0.01f;
  }
  DeviceBuffer<u64> d_handles(kCandidateCapacity);
  DeviceBuffer<f32> d_distances(kCandidateCapacity);
  DeviceBuffer<CycleResult> d_result(1);
  upload(d_handles, handles);
  upload(d_distances, distances);
  dominance_envelope_cycle_kernel<<<1, 128>>>(
    d_handles.get(), d_distances.get(), candidate_count,
    commit_capacity, issue_capacity, d_result.get());
  check_cuda(cudaGetLastError(), "cycle kernel launch");
  check_cuda(cudaDeviceSynchronize(), "cycle kernel synchronize");
  const CycleResult result = download(d_result, 1)[0];
  const u32 expected_hit = overflow ? 0u : 1u;
  if (result.mismatch != 0 ||
      result.envelope_hit != expected_hit ||
      result.certificate_count != result.oracle_count ||
      (result.envelope_hit != 0 &&
       (result.envelope_size < commit_capacity ||
        result.envelope_size > kPersistentFrontierRobCapacity ||
        result.certificate_count !=
          min(issue_capacity, result.envelope_size)))) {
    throw std::runtime_error(
      "cycle result mismatch issue=" + std::to_string(issue_capacity) +
      " overflow=" + std::to_string(overflow) +
      " hit=" + std::to_string(result.envelope_hit) +
      " envelope=" + std::to_string(result.envelope_size) +
      " mismatch=" + std::to_string(result.mismatch));
  }
  const double average_cycles =
    static_cast<double>(result.cycles) / kMeasuredIterations;
  std::cout
    << "dominance_envelope_cycle"
    << " commit=" << commit_capacity
    << " issue=" << issue_capacity
    << " path=" << (overflow ? "fallback" : "hit")
    << " envelope=" << result.envelope_size
    << " cycles=" << std::fixed << std::setprecision(1)
    << average_cycles
    << " us~=" << std::setprecision(3)
    << average_cycles / cycles_per_microsecond
    << '\n';
}

void run_cycle_microbenchmark(double cycles_per_microsecond) {
  for (const auto widths :
       {std::pair<u32, u32>{8u, 16u},
        std::pair<u32, u32>{16u, 32u}}) {
    run_cycle_case(
      false, widths.first, widths.second, cycles_per_microsecond);
    run_cycle_case(
      true, widths.first, widths.second, cycles_per_microsecond);
  }
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
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(
      cudaGetDeviceProperties(&properties, device),
      "cudaGetDeviceProperties");
    run_equivalence_cases();
    run_cycle_microbenchmark(
      static_cast<double>(properties.clockRate) / 1000.0);
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
