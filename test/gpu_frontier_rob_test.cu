#include <cuda_runtime.h>

#include <cstring>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "gpu_search/persistent_kernel/query_traversal.cuh"

namespace {

using gpu_search::DirectBatchPriority;
using gpu_search::FrontierRequestState;
using gpu_search::FrontierRobEntry;
using gpu_search::FrontierValidationState;
using gpu_search::PersistentKernelParams;
using gpu_search::DirectBatchDescriptor;
using gpu_search::kInvalidDeviceHandle;
using gpu_search::kFrontierRobFlagEarlyShadow;
using gpu_search::kPersistentFrontierRobCapacity;
using gpu_search::u8;
using gpu_search::u16;
using gpu_search::u32;
using gpu_search::u64;
using namespace gpu_search::persistent_kernel_detail;

struct RobTestResult {
  u32 first_issue_count{};
  u32 first_issue_span{};
  u32 first_controller_width{};
  u32 first_critical_misses{};
  u32 first_core_promoted{};
  u32 second_issue_count{};
  u32 second_issue_span{};
  u32 second_controller_width{};
  u32 second_critical_misses{};
  u32 second_speculative_promoted{};
  u32 second_controller_promoted{};
  u32 retained_issue_count{};
  u32 retained_issue_span{};
  u32 retained_controller_width{};
  u32 third_issue_count{};
  u32 third_issue_span{};
  u32 third_controller_width{};
  u32 failed{};
};

struct RobFastPathCycleResult {
  u64 prepare_cycles{};
  u64 plan_cycles{};
  u32 iterations{};
  u32 failed{};
};

struct EarlyShadowResult {
  u32 first_issue_count{};
  u32 first_issue_epoch{};
  u32 repeat_issue_count{};
  u32 repeat_issue_epoch{};
  u64 first_handle{};
  u64 last_handle{};
  u16 first_rank{};
  u16 last_rank{};
  u32 failed{};
};

struct LogicalHoleResult {
  u32 issue_count{};
  u32 physical_issue_span{};
  u32 issue_epochs{};
  u64 admitted_issue_width_sum{};
  u64 issue_width_capacity_sum{};
  u32 observed_admitted_width{};
  u32 hole_mapping{};
  u32 retained_mapping{};
  u32 failed{};
};

inline constexpr u32 kCertifiedSelectedCount = 6;
inline constexpr u32 kCertifiedMissCount = 4;

struct CertifiedReconcileResult {
  u32 failed{};
  u32 critical_fetch_count{};
  u32 critical_rob_hits{};
  u32 critical_misses{};
  u32 speculative_promoted{};
  u32 core_prefetch_promoted{};
  u32 core_prefetch_stale{};
  u32 feedback_promoted{};
  u32 feedback_core_hits{};
  u32 feedback_core_misses{};
  u32 feedback_retained{};
  u32 feedback_stale{};
  u32 feedback_queue_rejects{};
  u32 commit_rob_slots[kCertifiedSelectedCount]{};
  u32 graph_record_slots[kCertifiedSelectedCount]{};
  u64 critical_fetch_handles[kCertifiedMissCount]{};
  u32 critical_fetch_to_commit[kCertifiedMissCount]{};
  u8 mapped_states[5]{};
};

inline constexpr u32 kCriticalReservationCount = 5;
inline constexpr u32 kProtectedReservationCount = 3;

struct CriticalReservationResult {
  u32 failed{};
  u32 destinations[kCriticalReservationCount]{};
  u32 graph_failed{};
  u32 speculative_stale{};
  u64 speculative_wasted_bytes{};
  u32 core_prefetch_stale{};
  u32 feedback_stale{};
  u32 feedback_promoted{};
  u32 feedback_retained{};
  u32 protected_destinations[kProtectedReservationCount]{};
  u32 protected_graph_failed{};
  u32 protected_speculative_stale{};
  u64 protected_speculative_wasted_bytes{};
  u32 protected_core_prefetch_stale{};
  u32 protected_feedback_stale{};
};

struct UnderhintForceFullResult {
  u32 failed{};
  u32 any_with_underhint{};
  u32 any_without_underhint{};
  u8 selected_force_full[4]{};
  u8 positional_force_full[4]{};
  u8 general_force_full[4]{};
  u8 certified_force_full[4]{};
};

struct OwnerValidationFixture {
  alignas(16) u8 records[4][32]{};
  u32 shards[kPersistentFrontierRobCapacity]{};
  u32 unrelated_shards[kPersistentFrontierRobCapacity]{};
  u64 offsets[kPersistentFrontierRobCapacity]{};
  u64 local_iovas[kPersistentFrontierRobCapacity]{};
  u64 handles[kPersistentFrontierRobCapacity]{};
  u32 bytes[kPersistentFrontierRobCapacity]{};
  u8 states[kPersistentFrontierRobCapacity]{};
};

struct DynamicUnknownValidationFixture {
  gpu_search::DeviceShardRegion shard{};
  alignas(16) u8 record[gpu_search::kPersistentGraphReadBytes]{};
  u32 arena_state{};
  u32 output[3]{};
};

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

__global__ void rob_test_kernel(
    RobTestResult* output, bool physical_waste) {
  __shared__ FrontierRobEntry rob[kPersistentFrontierRobCapacity];
  __shared__ u64 preview_handles[kPersistentFrontierRobCapacity];
  __shared__ u16 preview_ranks[kPersistentFrontierRobCapacity];
  __shared__ u64 beam_handles[kPersistentFrontierRobCapacity];
  __shared__ u8 beam_expanded[kPersistentFrontierRobCapacity];
  __shared__ u64 selected_handles[kPersistentFrontierRobCapacity];
  __shared__ u32 selected_ranks[kPersistentFrontierRobCapacity];
  __shared__ u32 commit_slots[kPersistentFrontierRobCapacity];
  __shared__ u64 critical_handles[kPersistentFrontierRobCapacity];
  __shared__ u32 critical_to_commit[kPersistentFrontierRobCapacity];
  __shared__ u32 graph_slots[kPersistentFrontierRobCapacity];
  __shared__ u32 preview_count;
  __shared__ u32 selected_count;
  __shared__ u32 critical_count;
  __shared__ u32 issue_epoch;
  __shared__ u32 physical_issue_span;
  __shared__ u32 speculative_stale;
  __shared__ u64 speculative_wasted_bytes;
  __shared__ u32 issue_epochs;
  __shared__ u64 issue_width_sum;
  __shared__ u64 issue_capacity_sum;
  __shared__ u32 observed_issue_width;
  __shared__ u32 core_stale;
  __shared__ u32 shadow_count;
  __shared__ u32 speculative_promoted;
  __shared__ u32 core_promoted;
  __shared__ u32 critical_hits;
  __shared__ u32 critical_misses;
  __shared__ u32 commit_epochs;
  __shared__ u64 commit_width_sum;
  __shared__ u32 max_commit_width;
  __shared__ gpu_search::adaptive_frontier::ControllerState controller;
  __shared__ TailFrontierFeedback tail_feedback;

  for (u32 lane = threadIdx.x; lane < kPersistentFrontierRobCapacity;
       lane += blockDim.x) {
    rob[lane] = {};
    preview_handles[lane] = 100 + lane;
    preview_ranks[lane] = static_cast<u16>(lane);
    beam_handles[lane] = 100 + lane;
    beam_expanded[lane] = 0;
  }
  if (threadIdx.x == 0) {
    preview_count = kPersistentFrontierRobCapacity;
    selected_count = 0;
    critical_count = 0;
    issue_epoch = 0;
    physical_issue_span = 0;
    speculative_stale = 0;
    speculative_wasted_bytes = 0;
    issue_epochs = 0;
    issue_width_sum = 0;
    issue_capacity_sum = 0;
    observed_issue_width = 0;
    core_stale = 0;
    shadow_count = 0;
    speculative_promoted = 0;
    core_promoted = 0;
    critical_hits = 0;
    critical_misses = 0;
    commit_epochs = 0;
    commit_width_sum = 0;
    max_commit_width = 0;
    controller =
      gpu_search::adaptive_frontier::make_controller_state(16, 32);
    tail_feedback = {};
    *output = {};
  }
  __syncthreads();

  prepare_issue_frontier_entries(
    preview_handles, preview_ranks, preview_count, rob, issue_epoch,
    controller, 16, tail_feedback, speculative_stale,
    speculative_wasted_bytes, core_stale, issue_epochs,
    issue_width_sum, issue_capacity_sum, observed_issue_width,
    commit_slots, physical_issue_span);
  if (threadIdx.x == 0) {
    output->first_issue_count = preview_count;
    output->first_issue_span = physical_issue_span;
    output->first_controller_width = controller.current_issue_width;
  }
  for (u32 lane = threadIdx.x; lane < 17; lane += blockDim.x) {
    if (rob[lane].state !=
          static_cast<u8>(FrontierRequestState::issued) ||
        rob[lane].priority !=
          static_cast<u8>(
            lane < 16 ? DirectBatchPriority::critical
                      : DirectBatchPriority::speculative)) {
      atomicExch(&output->failed, 1u);
    }
    rob[lane].state = static_cast<u8>(FrontierRequestState::validated);
    rob[lane].transfer_bytes = 64;
  }
  __syncthreads();

  plan_commit_frontier(
    beam_handles, beam_expanded, 17, 16, rob, controller, true, 0, 0,
    selected_handles, selected_ranks, commit_slots, selected_count,
    critical_handles, critical_to_commit, critical_count, graph_slots,
    shadow_count, speculative_stale, speculative_wasted_bytes,
    speculative_promoted, core_stale, core_promoted, critical_hits,
    critical_misses, commit_epochs, commit_width_sum, max_commit_width,
    tail_feedback);
  if (threadIdx.x == 0) {
    output->first_critical_misses = critical_count;
    output->first_core_promoted = core_promoted;
  }
  for (u32 lane = threadIdx.x; lane < kPersistentFrontierRobCapacity;
       lane += blockDim.x) {
    if (rob[lane].state ==
        static_cast<u8>(FrontierRequestState::committed)) {
      rob[lane] = {};
    }
    if (lane < 16) {
      preview_handles[lane] = 116 + lane;
    } else {
      preview_handles[lane] = 200 + lane - 16;
    }
    preview_ranks[lane] = static_cast<u16>(lane);
  }
  if (threadIdx.x == 0) preview_count = kPersistentFrontierRobCapacity;
  __syncthreads();

  prepare_issue_frontier_entries(
    preview_handles, preview_ranks, preview_count, rob, issue_epoch,
    controller, 16, tail_feedback, speculative_stale,
    speculative_wasted_bytes, core_stale, issue_epochs,
    issue_width_sum, issue_capacity_sum, observed_issue_width,
    commit_slots, physical_issue_span);
  if (threadIdx.x == 0) {
    output->second_issue_count = preview_count;
    output->second_issue_span = physical_issue_span;
    output->second_controller_width = controller.current_issue_width;
    // Logical position zero is the promoted record retained in physical slot
    // 16. The last logical position must therefore be issued in slot 17. A
    // descriptor bounded by the logical count (17) would silently omit it.
    if (commit_slots[0] != 16 || commit_slots[16] != 17 ||
        physical_issue_span != 18 ||
        rob[15].state != static_cast<u8>(FrontierRequestState::init) ||
        rob[17].state != static_cast<u8>(FrontierRequestState::issued)) {
      output->failed |= 256u;
    }
  }
  if (physical_waste) {
    for (u32 lane = threadIdx.x;
         lane < kPersistentFrontierRobCapacity; lane += blockDim.x) {
      if (lane < 16) {
        rob[lane] = {};
      } else if (rob[lane].state !=
                 static_cast<u8>(FrontierRequestState::init)) {
        rob[lane].state =
          static_cast<u8>(FrontierRequestState::validated);
        rob[lane].transfer_bytes = 64;
      }
      preview_handles[lane] = 300 + lane;
      preview_ranks[lane] = static_cast<u16>(lane);
    }
    if (threadIdx.x == 0) {
      preview_count = kPersistentFrontierRobCapacity;
      tail_feedback = {};
      tail_feedback.stale = 1;
    }
    __syncthreads();
    prepare_issue_frontier_entries(
      preview_handles, preview_ranks, preview_count, rob, issue_epoch,
      controller, 16, tail_feedback, speculative_stale,
      speculative_wasted_bytes, core_stale, issue_epochs,
      issue_width_sum, issue_capacity_sum, observed_issue_width,
      commit_slots, physical_issue_span);
    if (threadIdx.x == 0) {
      output->third_issue_count = preview_count;
      output->third_issue_span = physical_issue_span;
      output->third_controller_width = controller.current_issue_width;
    }
    return;
  }
  for (u32 lane = threadIdx.x; lane < kPersistentFrontierRobCapacity;
       lane += blockDim.x) {
    if (rob[lane].state ==
        static_cast<u8>(FrontierRequestState::issued)) {
      rob[lane].state = static_cast<u8>(FrontierRequestState::validated);
      rob[lane].transfer_bytes = 64;
    }
    if (lane < 16) {
      beam_handles[lane] =
        lane == 0 ? 200 : 116 + lane;
      beam_expanded[lane] = 0;
    }
  }
  __syncthreads();

  // A validated shadow entry that remains in the exact preview is stability
  // evidence exactly once for that physical request. Reconciliation may run
  // repeatedly before commit, but the utility-accounted flag prevents the
  // same resident record from manufacturing another retention sample or
  // growing the controller twice.
  if (threadIdx.x == 0) {
    preview_count = kPersistentFrontierRobCapacity;
  }
  __syncthreads();
  prepare_issue_frontier_entries(
    preview_handles, preview_ranks, preview_count, rob, issue_epoch,
    controller, 16, tail_feedback, speculative_stale,
    speculative_wasted_bytes, core_stale, issue_epochs,
    issue_width_sum, issue_capacity_sum, observed_issue_width,
    commit_slots, physical_issue_span);
  if (threadIdx.x == 0) {
    preview_count = kPersistentFrontierRobCapacity;
  }
  __syncthreads();
  prepare_issue_frontier_entries(
    preview_handles, preview_ranks, preview_count, rob, issue_epoch,
    controller, 16, tail_feedback, speculative_stale,
    speculative_wasted_bytes, core_stale, issue_epochs,
    issue_width_sum, issue_capacity_sum, observed_issue_width,
    commit_slots, physical_issue_span);
  if (threadIdx.x == 0) {
    output->retained_issue_count = preview_count;
    output->retained_issue_span = physical_issue_span;
    output->retained_controller_width = controller.current_issue_width;
  }
  __syncthreads();

  plan_commit_frontier(
    beam_handles, beam_expanded, 16, 16, rob, controller, true, 0, 0,
    selected_handles, selected_ranks, commit_slots, selected_count,
    critical_handles, critical_to_commit, critical_count, graph_slots,
    shadow_count, speculative_stale, speculative_wasted_bytes,
    speculative_promoted, core_stale, core_promoted, critical_hits,
    critical_misses, commit_epochs, commit_width_sum, max_commit_width,
    tail_feedback);
  if (threadIdx.x == 0) {
    output->second_critical_misses = critical_count;
    output->second_speculative_promoted = speculative_promoted;
    output->second_controller_promoted = tail_feedback.promoted;
  }
}

__global__ void early_shadow_test_kernel(EarlyShadowResult* output) {
  constexpr u32 core_slot_count = 16;
  constexpr u32 beam_count = 24;
  constexpr u32 commit_count = 4;
  constexpr u32 requested_count = 3;
  constexpr u32 initial_epoch = 9;
  __shared__ FrontierRobEntry rob[kPersistentFrontierRobCapacity];
  __shared__ u64 beam_handles[kPersistentFrontierRobCapacity];
  __shared__ u8 beam_expanded[kPersistentFrontierRobCapacity];
  __shared__ u32 issue_count;
  __shared__ u32 issue_epoch;

  for (u32 lane = threadIdx.x; lane < kPersistentFrontierRobCapacity;
       lane += blockDim.x) {
    rob[lane] = {};
    beam_handles[lane] = 1000 + lane;
    beam_expanded[lane] =
      static_cast<u8>(lane == 1 || lane == 4);
    if (lane < core_slot_count) {
      FrontierRobEntry& entry = rob[lane];
      entry.node_handle = 9000 + lane;
      entry.issue_epoch = 70 + lane;
      entry.transfer_bytes = 200 + lane;
      entry.beam_rank = static_cast<u16>(400 + lane);
      entry.scratch_slot = static_cast<u8>(lane);
      entry.state = static_cast<u8>(FrontierRequestState::inflight);
      entry.validation =
        static_cast<u8>(FrontierValidationState::valid);
      entry.priority =
        static_cast<u8>(DirectBatchPriority::critical);
    }
  }
  if (threadIdx.x == 0) {
    issue_count = 0;
    issue_epoch = initial_epoch;
    *output = {};
  }
  __syncthreads();

  prepare_early_shadow_frontier(
    beam_handles, beam_expanded, beam_count, commit_count,
    requested_count, rob, core_slot_count, issue_count, issue_epoch);
  if (threadIdx.x == 0) {
    output->first_issue_count = issue_count;
    output->first_issue_epoch = issue_epoch;
    output->first_handle = rob[core_slot_count].node_handle;
    output->last_handle =
      rob[core_slot_count + requested_count - 1].node_handle;
    output->first_rank = rob[core_slot_count].beam_rank;
    output->last_rank =
      rob[core_slot_count + requested_count - 1].beam_rank;
    if (issue_count != requested_count ||
        issue_epoch != initial_epoch + 1) {
      atomicOr(&output->failed, 1u);
    }
  }
  if (threadIdx.x < core_slot_count) {
    const u32 lane = threadIdx.x;
    const FrontierRobEntry& entry = rob[lane];
    if (entry.node_handle != 9000 + lane ||
        entry.issue_epoch != 70 + lane ||
        entry.transfer_bytes != 200 + lane ||
        entry.beam_rank != static_cast<u16>(400 + lane) ||
        entry.scratch_slot != static_cast<u8>(lane) ||
        entry.state !=
          static_cast<u8>(FrontierRequestState::inflight) ||
        entry.validation !=
          static_cast<u8>(FrontierValidationState::valid) ||
        entry.priority !=
          static_cast<u8>(DirectBatchPriority::critical)) {
      atomicOr(&output->failed, 2u);
    }
  }
  if (threadIdx.x < requested_count) {
    const u32 ordinal = threadIdx.x;
    const u32 slot = core_slot_count + ordinal;
    const u32 rank = 6 + ordinal;
    const FrontierRobEntry& entry = rob[slot];
    if (entry.node_handle != beam_handles[rank] ||
        entry.issue_epoch != initial_epoch + 1 ||
        entry.beam_rank != rank ||
        entry.scratch_slot != slot ||
        entry.flags != kFrontierRobFlagEarlyShadow ||
        entry.state !=
          static_cast<u8>(FrontierRequestState::issued) ||
        entry.priority !=
          static_cast<u8>(DirectBatchPriority::speculative)) {
      atomicOr(&output->failed, 4u);
    }
  }
  if (threadIdx.x >= core_slot_count + requested_count &&
      threadIdx.x < kPersistentFrontierRobCapacity) {
    const FrontierRobEntry& entry = rob[threadIdx.x];
    if (entry.state != static_cast<u8>(FrontierRequestState::init) ||
        entry.node_handle != kInvalidDeviceHandle) {
      atomicOr(&output->failed, 8u);
    }
  }
  __syncthreads();

  // ISSUED tail metadata owns its scratch slots. A second look-ahead attempt
  // must observe the resident tail, return no work, and leave both regions
  // byte-for-byte logically unchanged.
  prepare_early_shadow_frontier(
    beam_handles, beam_expanded, beam_count, commit_count,
    requested_count, rob, core_slot_count, issue_count, issue_epoch);
  if (threadIdx.x == 0) {
    output->repeat_issue_count = issue_count;
    output->repeat_issue_epoch = issue_epoch;
    if (issue_count != 0 || issue_epoch != initial_epoch + 1) {
      atomicOr(&output->failed, 16u);
    }
  }
  if (threadIdx.x < core_slot_count) {
    const u32 lane = threadIdx.x;
    const FrontierRobEntry& entry = rob[lane];
    if (entry.node_handle != 9000 + lane ||
        entry.issue_epoch != 70 + lane ||
        entry.transfer_bytes != 200 + lane ||
        entry.beam_rank != static_cast<u16>(400 + lane) ||
        entry.scratch_slot != static_cast<u8>(lane) ||
        entry.state !=
          static_cast<u8>(FrontierRequestState::inflight) ||
        entry.validation !=
          static_cast<u8>(FrontierValidationState::valid) ||
        entry.priority !=
          static_cast<u8>(DirectBatchPriority::critical)) {
      atomicOr(&output->failed, 32u);
    }
  } else if (threadIdx.x < core_slot_count + requested_count) {
    const u32 ordinal = threadIdx.x - core_slot_count;
    const u32 rank = 6 + ordinal;
    const FrontierRobEntry& entry = rob[threadIdx.x];
    if (entry.node_handle != beam_handles[rank] ||
        entry.issue_epoch != initial_epoch + 1 ||
        entry.beam_rank != rank ||
        entry.scratch_slot != threadIdx.x ||
        entry.flags != kFrontierRobFlagEarlyShadow ||
        entry.state !=
          static_cast<u8>(FrontierRequestState::issued) ||
        entry.priority !=
          static_cast<u8>(DirectBatchPriority::speculative)) {
      atomicOr(&output->failed, 64u);
    }
  } else if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const FrontierRobEntry& entry = rob[threadIdx.x];
    if (entry.state != static_cast<u8>(FrontierRequestState::init) ||
        entry.node_handle != kInvalidDeviceHandle) {
      atomicOr(&output->failed, 128u);
    }
  }
}

__global__ void logical_hole_test_kernel(LogicalHoleResult* output) {
  constexpr u32 core_slot_count = 16;
  constexpr u32 certificate_count = 18;
  constexpr u32 hole_position = 16;
  constexpr u32 retained_position = 17;
  constexpr u32 retained_slot = kPersistentFrontierRobCapacity - 1;
  constexpr u32 initial_issue_epoch = 40;
  constexpr u32 resident_issue_epoch = 31;

  __shared__ FrontierRobEntry rob[kPersistentFrontierRobCapacity];
  __shared__ u64 preview_handles[kPersistentFrontierRobCapacity];
  __shared__ u16 preview_ranks[kPersistentFrontierRobCapacity];
  __shared__ u32 issue_rob_slots[kPersistentFrontierRobCapacity];
  __shared__ u32 issue_count;
  __shared__ u32 issue_epoch;
  __shared__ u32 physical_issue_span;
  __shared__ u32 speculative_stale;
  __shared__ u64 speculative_wasted_bytes;
  __shared__ u32 core_prefetch_stale;
  __shared__ u32 issue_epochs;
  __shared__ u64 issue_width_sum;
  __shared__ u64 issue_width_capacity_sum;
  __shared__ u32 observed_max_issue_width;
  __shared__ gpu_search::adaptive_frontier::ControllerState controller;
  __shared__ TailFrontierFeedback feedback;

  const u32 lane = threadIdx.x;
  if (lane < kPersistentFrontierRobCapacity) {
    rob[lane] = {};
    preview_handles[lane] = 7000 + lane;
    preview_ranks[lane] = static_cast<u16>(lane);
    issue_rob_slots[lane] = 0xdeadbeefu;
    if (lane >= core_slot_count) {
      FrontierRobEntry& entry = rob[lane];
      entry.node_handle = 9000 + lane;
      entry.issue_epoch = resident_issue_epoch;
      entry.beam_rank = static_cast<u16>(300 + lane);
      entry.scratch_slot = static_cast<u8>(lane);
      entry.state = static_cast<u8>(
        (lane & 1u) != 0
          ? FrontierRequestState::inflight
          : FrontierRequestState::arrived);
      entry.priority =
        static_cast<u8>(DirectBatchPriority::speculative);
    }
  }
  if (lane == 0) {
    // No tail slot is free. Logical position 16 deliberately misses every
    // resident request, while position 17 matches the last physical slot.
    // The admitted set is therefore {0..15,17}: it is not a logical prefix.
    rob[retained_slot].node_handle = preview_handles[retained_position];
    issue_count = certificate_count;
    issue_epoch = initial_issue_epoch;
    physical_issue_span = 0;
    speculative_stale = 0;
    speculative_wasted_bytes = 0;
    core_prefetch_stale = 0;
    issue_epochs = 0;
    issue_width_sum = 0;
    issue_width_capacity_sum = 0;
    observed_max_issue_width = 0;
    controller =
      gpu_search::adaptive_frontier::make_controller_state(
        core_slot_count, kPersistentFrontierRobCapacity);
    controller.current_issue_width = certificate_count;
    feedback = {};
    *output = {};
  }
  __syncthreads();

  prepare_issue_frontier_entries(
    preview_handles, preview_ranks, issue_count, rob, issue_epoch,
    controller, core_slot_count, feedback, speculative_stale,
    speculative_wasted_bytes, core_prefetch_stale, issue_epochs,
    issue_width_sum, issue_width_capacity_sum, observed_max_issue_width,
    issue_rob_slots, physical_issue_span,
    /*allow_new_tail=*/false,
    /*apply_controller_feedback=*/false,
    /*start_new_issue_epoch=*/true);

  if (lane < core_slot_count) {
    const FrontierRobEntry& entry = rob[lane];
    if (issue_rob_slots[lane] != lane ||
        entry.node_handle != preview_handles[lane] ||
        entry.state !=
          static_cast<u8>(FrontierRequestState::issued) ||
        entry.priority !=
          static_cast<u8>(DirectBatchPriority::critical)) {
      atomicOr(&output->failed, 1u);
    }
  } else if (lane < kPersistentFrontierRobCapacity) {
    // Reconciliation must not overwrite an in-flight/arrived tail merely
    // because it did not match the new exact certificate.
    const FrontierRobEntry& entry = rob[lane];
    if (entry.issue_epoch != resident_issue_epoch ||
        entry.scratch_slot != lane ||
        entry.priority !=
          static_cast<u8>(DirectBatchPriority::speculative) ||
        entry.state != static_cast<u8>(
          (lane & 1u) != 0
            ? FrontierRequestState::inflight
            : FrontierRequestState::arrived)) {
      atomicOr(&output->failed, 2u);
    }
  }
  __syncthreads();

  if (lane == 0) {
    output->issue_count = issue_count;
    output->physical_issue_span = physical_issue_span;
    output->issue_epochs = issue_epochs;
    output->admitted_issue_width_sum = issue_width_sum;
    output->issue_width_capacity_sum = issue_width_capacity_sum;
    output->observed_admitted_width = observed_max_issue_width;
    output->hole_mapping = issue_rob_slots[hole_position];
    output->retained_mapping = issue_rob_slots[retained_position];
    if (issue_count != certificate_count ||
        issue_rob_slots[hole_position] != UINT32_MAX ||
        issue_rob_slots[retained_position] != retained_slot ||
        physical_issue_span != kPersistentFrontierRobCapacity ||
        issue_epochs != 1 || issue_width_sum != certificate_count - 1 ||
        issue_width_capacity_sum != certificate_count ||
        observed_max_issue_width != certificate_count - 1 ||
        issue_epoch != initial_issue_epoch + 1) {
      output->failed |= 4u;
    }
  }
}

__global__ void certified_reconcile_test_kernel(
    CertifiedReconcileResult* output, bool issue_wave) {
  constexpr u32 speculative_slot = 23;
  constexpr u32 critical_slot = 4;
  constexpr u32 inflight_slot = 29;
  constexpr u32 mismatched_core_slot = 7;
  constexpr u32 mismatched_speculative_slot = 12;
  constexpr u32 invalid_slot = kPersistentFrontierRobCapacity + 5;
  constexpr u64 first_selected_handle = 5000;
  constexpr u32 initial_critical_rob_hits = 10;
  constexpr u32 initial_critical_misses = 20;
  constexpr u32 initial_speculative_promoted = 30;
  constexpr u32 initial_core_prefetch_promoted = 40;
  constexpr u32 initial_core_prefetch_stale = 50;
  constexpr u32 initial_feedback_promoted = 60;
  constexpr u32 initial_feedback_retained = 61;
  constexpr u32 initial_feedback_stale = 62;
  constexpr u32 initial_feedback_queue_rejects = 63;
  constexpr u32 initial_feedback_core_hits = 70;
  constexpr u32 initial_feedback_core_misses = 80;

  __shared__ FrontierRobEntry rob[kPersistentFrontierRobCapacity];
  __shared__ u64 selected_handles[kPersistentFrontierRobCapacity];
  __shared__ u32 certified_slots[kPersistentFrontierRobCapacity];
  __shared__ u32 commit_slots[kPersistentFrontierRobCapacity];
  __shared__ u32 graph_slots[kPersistentFrontierRobCapacity];
  __shared__ u64 critical_handles[kPersistentFrontierRobCapacity];
  __shared__ u32 critical_to_commit[kPersistentFrontierRobCapacity];
  __shared__ u32 critical_count;
  __shared__ u32 critical_rob_hits;
  __shared__ u32 critical_misses;
  __shared__ u32 speculative_promoted;
  __shared__ u32 core_prefetch_promoted;
  __shared__ u32 core_prefetch_stale;
  __shared__ u32 failed;
  __shared__ TailFrontierFeedback feedback;
  __shared__ CertifiedCommitReconcileContext context;

  const u32 lane = threadIdx.x;
  if (lane < kPersistentFrontierRobCapacity) {
    rob[lane] = {};
    selected_handles[lane] = kInvalidDeviceHandle;
    certified_slots[lane] = UINT32_MAX;
    commit_slots[lane] = 0xdeadbeefu;
    graph_slots[lane] = 0xdeadbeefu;
    critical_handles[lane] = kInvalidDeviceHandle;
    critical_to_commit[lane] = UINT32_MAX;
  }
  if (lane == 0) {
    critical_count = UINT32_MAX;
    critical_rob_hits = initial_critical_rob_hits;
    critical_misses = initial_critical_misses;
    speculative_promoted = initial_speculative_promoted;
    core_prefetch_promoted = initial_core_prefetch_promoted;
    core_prefetch_stale = initial_core_prefetch_stale;
    failed = 0;
    feedback = {};
    feedback.promoted = initial_feedback_promoted;
    feedback.retained = initial_feedback_retained;
    feedback.stale = initial_feedback_stale;
    feedback.queue_rejects = initial_feedback_queue_rejects;
    feedback.core_hits = initial_feedback_core_hits;
    feedback.core_misses = initial_feedback_core_misses;
    *output = {};
  }
  __syncthreads();

  if (lane == 0) {
    for (u32 position = 0; position < kCertifiedSelectedCount;
         ++position) {
      selected_handles[position] = first_selected_handle + position;
    }
    certified_slots[0] = speculative_slot;
    certified_slots[1] = critical_slot;
    certified_slots[2] = inflight_slot;
    certified_slots[3] = mismatched_core_slot;
    certified_slots[4] = invalid_slot;
    certified_slots[5] = mismatched_speculative_slot;

    rob[speculative_slot].node_handle = selected_handles[0];
    rob[speculative_slot].scratch_slot = 9;
    rob[speculative_slot].state =
      static_cast<u8>(FrontierRequestState::validated);
    rob[speculative_slot].validation =
      static_cast<u8>(FrontierValidationState::valid);
    rob[speculative_slot].priority =
      static_cast<u8>(DirectBatchPriority::speculative);

    rob[critical_slot].node_handle = selected_handles[1];
    rob[critical_slot].scratch_slot = 27;
    rob[critical_slot].state =
      static_cast<u8>(FrontierRequestState::validated);
    rob[critical_slot].validation =
      static_cast<u8>(FrontierValidationState::valid);
    rob[critical_slot].priority =
      static_cast<u8>(DirectBatchPriority::critical);

    rob[inflight_slot].node_handle = selected_handles[2];
    rob[inflight_slot].scratch_slot = 15;
    rob[inflight_slot].state =
      static_cast<u8>(FrontierRequestState::inflight);
    rob[inflight_slot].validation =
      static_cast<u8>(FrontierValidationState::unknown);
    rob[inflight_slot].priority =
      static_cast<u8>(DirectBatchPriority::critical);

    rob[mismatched_core_slot].node_handle = 9003;
    rob[mismatched_core_slot].scratch_slot = 3;
    rob[mismatched_core_slot].state =
      static_cast<u8>(FrontierRequestState::validated);
    rob[mismatched_core_slot].validation =
      static_cast<u8>(FrontierValidationState::valid);
    rob[mismatched_core_slot].priority =
      static_cast<u8>(DirectBatchPriority::critical);

    rob[mismatched_speculative_slot].node_handle = 9005;
    rob[mismatched_speculative_slot].scratch_slot = 5;
    rob[mismatched_speculative_slot].state =
      static_cast<u8>(FrontierRequestState::validated);
    rob[mismatched_speculative_slot].validation =
      static_cast<u8>(FrontierValidationState::valid);
    rob[mismatched_speculative_slot].priority =
      static_cast<u8>(DirectBatchPriority::speculative);

    context = CertifiedCommitReconcileContext{
      .selected_handles = selected_handles,
      .certified_rob_slots = certified_slots,
      .frontier_rob = rob,
      .commit_rob_slots = commit_slots,
      .graph_record_slots = graph_slots,
      .critical_fetch_handles = critical_handles,
      .critical_fetch_to_commit = critical_to_commit,
      .critical_fetch_count = &critical_count,
      .critical_rob_hits = &critical_rob_hits,
      .critical_misses = &critical_misses,
      .speculative_promoted = &speculative_promoted,
      .core_prefetch_promoted = &core_prefetch_promoted,
      .core_prefetch_stale = &core_prefetch_stale,
      .feedback = &feedback,
    };
  }
  __syncthreads();

  reconcile_certified_commit_frontier(
    context, kCertifiedSelectedCount, issue_wave);

  const u32 expected_commit_slot =
    lane == 0 ? speculative_slot :
    lane == 1 ? critical_slot : UINT32_MAX;
  if (commit_slots[lane] != expected_commit_slot) {
    atomicOr(&failed, 1u);
  }
  if (lane < kCertifiedSelectedCount) {
    const u32 expected_graph_slot =
      lane == 0 ? kGraphScratchBit | 9u :
      lane == 1 ? kGraphScratchBit | 27u : UINT32_MAX;
    if (graph_slots[lane] != expected_graph_slot) {
      atomicOr(&failed, 2u);
    }
  }
  __syncthreads();

  if (lane == 0) {
    constexpr u64 expected_miss_handles[kCertifiedMissCount]{
      first_selected_handle + 2,
      first_selected_handle + 3,
      first_selected_handle + 4,
      first_selected_handle + 5,
    };
    constexpr u32 expected_miss_positions[kCertifiedMissCount]{2, 3, 4, 5};
    for (u32 index = 0; index < kCertifiedMissCount; ++index) {
      if (critical_handles[index] != expected_miss_handles[index] ||
          critical_to_commit[index] != expected_miss_positions[index]) {
        failed |= 4u;
      }
    }
    if (rob[speculative_slot].state !=
          static_cast<u8>(FrontierRequestState::committed) ||
        rob[critical_slot].state !=
          static_cast<u8>(FrontierRequestState::committed) ||
        rob[inflight_slot].state !=
          static_cast<u8>(FrontierRequestState::inflight) ||
        rob[mismatched_core_slot].state !=
          static_cast<u8>(FrontierRequestState::validated) ||
        rob[mismatched_speculative_slot].state !=
          static_cast<u8>(FrontierRequestState::validated)) {
      failed |= 8u;
    }
    if (critical_count != kCertifiedMissCount ||
        critical_rob_hits != initial_critical_rob_hits + 2 ||
        critical_misses !=
          initial_critical_misses + kCertifiedMissCount ||
        speculative_promoted != initial_speculative_promoted + 1 ||
        core_prefetch_promoted != initial_core_prefetch_promoted + 1 ||
        core_prefetch_stale != initial_core_prefetch_stale + 1) {
      failed |= 16u;
    }
    if (feedback.promoted != initial_feedback_promoted + 1 ||
        feedback.core_hits !=
          initial_feedback_core_hits + (issue_wave ? 2u : 0u) ||
        feedback.core_misses !=
          initial_feedback_core_misses +
            (issue_wave ? kCertifiedMissCount : 0u) ||
        feedback.retained != initial_feedback_retained ||
        feedback.stale != initial_feedback_stale ||
        feedback.queue_rejects != initial_feedback_queue_rejects) {
      failed |= 32u;
    }

    output->failed = failed;
    output->critical_fetch_count = critical_count;
    output->critical_rob_hits = critical_rob_hits;
    output->critical_misses = critical_misses;
    output->speculative_promoted = speculative_promoted;
    output->core_prefetch_promoted = core_prefetch_promoted;
    output->core_prefetch_stale = core_prefetch_stale;
    output->feedback_promoted = feedback.promoted;
    output->feedback_core_hits = feedback.core_hits;
    output->feedback_core_misses = feedback.core_misses;
    output->feedback_retained = feedback.retained;
    output->feedback_stale = feedback.stale;
    output->feedback_queue_rejects = feedback.queue_rejects;
    for (u32 position = 0; position < kCertifiedSelectedCount;
         ++position) {
      output->commit_rob_slots[position] = commit_slots[position];
      output->graph_record_slots[position] = graph_slots[position];
    }
    for (u32 index = 0; index < kCertifiedMissCount; ++index) {
      output->critical_fetch_handles[index] = critical_handles[index];
      output->critical_fetch_to_commit[index] =
        critical_to_commit[index];
    }
    output->mapped_states[0] = rob[speculative_slot].state;
    output->mapped_states[1] = rob[critical_slot].state;
    output->mapped_states[2] = rob[inflight_slot].state;
    output->mapped_states[3] = rob[mismatched_core_slot].state;
    output->mapped_states[4] = rob[mismatched_speculative_slot].state;
  }
}

__device__ bool reservation_entry_matches(
    const FrontierRobEntry& entry, u32 lane, FrontierRequestState state) {
  return entry.node_handle == 12000 + lane &&
         entry.issue_epoch == 300 + lane &&
         entry.transfer_bytes == 600 + lane &&
         entry.beam_rank == static_cast<u16>(700 + lane) &&
         entry.scratch_slot == static_cast<u8>(lane) &&
         entry.state == static_cast<u8>(state) &&
         entry.validation ==
           static_cast<u8>(FrontierValidationState::valid) &&
         entry.priority ==
           static_cast<u8>(DirectBatchPriority::critical) &&
         entry.flags == static_cast<u8>(lane & 3u);
}

__device__ bool reservation_entry_is_reset(
    const FrontierRobEntry& entry) {
  return entry.node_handle == kInvalidDeviceHandle &&
         entry.issue_epoch == 0 && entry.transfer_bytes == 0 &&
         entry.beam_rank == 0 && entry.scratch_slot == 0 &&
         entry.state == static_cast<u8>(FrontierRequestState::init) &&
         entry.validation ==
           static_cast<u8>(FrontierValidationState::unknown) &&
         entry.priority ==
           static_cast<u8>(DirectBatchPriority::speculative) &&
         entry.flags == 0;
}

__global__ void critical_reservation_test_kernel(
    CriticalReservationResult* output) {
  constexpr u32 empty_slot = 29;
  constexpr u32 stale_slot = 30;
  constexpr u32 first_speculative_slot = 4;
  constexpr u32 second_speculative_slot = 27;
  constexpr u32 first_critical_slot = 2;
  constexpr u32 second_critical_slot = 26;
  constexpr u32 expected_destinations[kCriticalReservationCount]{
    empty_slot,
    stale_slot,
    first_speculative_slot,
    second_speculative_slot,
    first_critical_slot,
  };
  constexpr u32 initial_speculative_stale = 7;
  constexpr u64 initial_speculative_wasted_bytes = 1000;
  constexpr u32 initial_core_prefetch_stale = 11;
  constexpr u32 initial_feedback_stale = 13;
  constexpr u32 initial_feedback_promoted = 17;
  constexpr u32 initial_feedback_retained = 19;
  constexpr u32 first_speculative_bytes = 128;
  constexpr u32 second_speculative_bytes = 256;

  __shared__ FrontierRobEntry rob[kPersistentFrontierRobCapacity];
  __shared__ u32 destinations[kPersistentFrontierRobCapacity];
  __shared__ u32 graph_failed;
  __shared__ u32 speculative_stale;
  __shared__ u64 speculative_wasted_bytes;
  __shared__ u32 core_prefetch_stale;
  __shared__ TailFrontierFeedback feedback;

  const u32 lane = threadIdx.x;
  if (lane < kPersistentFrontierRobCapacity) {
    FrontierRobEntry& entry = rob[lane];
    entry.node_handle = 12000 + lane;
    entry.issue_epoch = 300 + lane;
    entry.transfer_bytes = 600 + lane;
    entry.beam_rank = static_cast<u16>(700 + lane);
    entry.scratch_slot = static_cast<u8>(lane);
    entry.state = static_cast<u8>(
      (lane & 1u) != 0 ? FrontierRequestState::inflight
                       : FrontierRequestState::committed);
    entry.validation =
      static_cast<u8>(FrontierValidationState::valid);
    entry.priority =
      static_cast<u8>(DirectBatchPriority::critical);
    entry.flags = static_cast<u8>(lane & 3u);
    destinations[lane] = 0xdeadbeefu;
  }
  __syncthreads();
  if (lane == 0) {
    rob[empty_slot] = {};
    rob[stale_slot].state =
      static_cast<u8>(FrontierRequestState::stale);
    rob[first_speculative_slot].state =
      static_cast<u8>(FrontierRequestState::validated);
    rob[first_speculative_slot].priority =
      static_cast<u8>(DirectBatchPriority::speculative);
    rob[first_speculative_slot].transfer_bytes =
      first_speculative_bytes;
    rob[second_speculative_slot].state =
      static_cast<u8>(FrontierRequestState::validated);
    rob[second_speculative_slot].priority =
      static_cast<u8>(DirectBatchPriority::speculative);
    rob[second_speculative_slot].transfer_bytes =
      second_speculative_bytes;
    rob[first_critical_slot].state =
      static_cast<u8>(FrontierRequestState::validated);
    rob[second_critical_slot].state =
      static_cast<u8>(FrontierRequestState::validated);
    graph_failed = 0;
    speculative_stale = initial_speculative_stale;
    speculative_wasted_bytes = initial_speculative_wasted_bytes;
    core_prefetch_stale = initial_core_prefetch_stale;
    feedback = {};
    feedback.stale = initial_feedback_stale;
    feedback.promoted = initial_feedback_promoted;
    feedback.retained = initial_feedback_retained;
    *output = {};
  }
  __syncthreads();

  reserve_critical_fetch_destinations(
    rob, kCriticalReservationCount, destinations, graph_failed,
    speculative_stale, speculative_wasted_bytes,
    core_prefetch_stale, feedback);

  if (lane < kCriticalReservationCount &&
      destinations[lane] != expected_destinations[lane]) {
    atomicOr(&output->failed, 1u);
  }
  if (lane < kCriticalReservationCount &&
      !reservation_entry_is_reset(rob[expected_destinations[lane]])) {
    atomicOr(&output->failed, 2u);
  }
  if (lane == 0) {
    // These four live states are never legal victims, even though they
    // precede every free slot in physical ROB order.
    if (!reservation_entry_matches(
          rob[0], 0, FrontierRequestState::committed) ||
        !reservation_entry_matches(
          rob[1], 1, FrontierRequestState::inflight) ||
        !reservation_entry_matches(
          rob[3], 3, FrontierRequestState::inflight) ||
        !reservation_entry_matches(
          rob[5], 5, FrontierRequestState::inflight) ||
        !reservation_entry_matches(
          rob[second_critical_slot], second_critical_slot,
          FrontierRequestState::validated)) {
      atomicOr(&output->failed, 4u);
    }
    if (graph_failed != 0 ||
        speculative_stale != initial_speculative_stale + 2 ||
        speculative_wasted_bytes !=
          initial_speculative_wasted_bytes +
            first_speculative_bytes + second_speculative_bytes ||
        core_prefetch_stale != initial_core_prefetch_stale + 1 ||
        feedback.stale != initial_feedback_stale + 2 ||
        feedback.promoted != initial_feedback_promoted ||
        feedback.retained != initial_feedback_retained) {
      atomicOr(&output->failed, 8u);
    }
    for (u32 index = 0; index < kCriticalReservationCount; ++index) {
      output->destinations[index] = destinations[index];
    }
    output->graph_failed = graph_failed;
    output->speculative_stale = speculative_stale;
    output->speculative_wasted_bytes = speculative_wasted_bytes;
    output->core_prefetch_stale = core_prefetch_stale;
    output->feedback_stale = feedback.stale;
    output->feedback_promoted = feedback.promoted;
    output->feedback_retained = feedback.retained;
  }
  __syncthreads();

  // With every slot live, critical allocation must fail without clearing
  // either COMMITTED records or requests still capable of a late DMA.
  if (lane < kPersistentFrontierRobCapacity) {
    FrontierRobEntry& entry = rob[lane];
    entry.node_handle = 12000 + lane;
    entry.issue_epoch = 300 + lane;
    entry.transfer_bytes = 600 + lane;
    entry.beam_rank = static_cast<u16>(700 + lane);
    entry.scratch_slot = static_cast<u8>(lane);
    const FrontierRequestState protected_state =
      (lane & 1u) != 0 ? FrontierRequestState::inflight
                       : FrontierRequestState::committed;
    entry.state = static_cast<u8>(protected_state);
    entry.validation =
      static_cast<u8>(FrontierValidationState::valid);
    entry.priority =
      static_cast<u8>(DirectBatchPriority::critical);
    entry.flags = static_cast<u8>(lane & 3u);
    destinations[lane] = 0xdeadbeefu;
  }
  if (lane == 0) {
    graph_failed = 0;
    speculative_stale = initial_speculative_stale;
    speculative_wasted_bytes = initial_speculative_wasted_bytes;
    core_prefetch_stale = initial_core_prefetch_stale;
    feedback = {};
    feedback.stale = initial_feedback_stale;
  }
  __syncthreads();

  reserve_critical_fetch_destinations(
    rob, kProtectedReservationCount, destinations, graph_failed,
    speculative_stale, speculative_wasted_bytes,
    core_prefetch_stale, feedback);

  if (lane < kPersistentFrontierRobCapacity) {
    const FrontierRequestState expected_state =
      (lane & 1u) != 0 ? FrontierRequestState::inflight
                       : FrontierRequestState::committed;
    if (!reservation_entry_matches(rob[lane], lane, expected_state)) {
      atomicOr(&output->failed, 16u);
    }
  }
  if (lane == 0) {
    if (destinations[0] != UINT32_MAX ||
        destinations[1] != UINT32_MAX ||
        destinations[2] != UINT32_MAX ||
        graph_failed != 5 ||
        speculative_stale != initial_speculative_stale ||
        speculative_wasted_bytes !=
          initial_speculative_wasted_bytes ||
        core_prefetch_stale != initial_core_prefetch_stale ||
        feedback.stale != initial_feedback_stale) {
      atomicOr(&output->failed, 32u);
    }
    for (u32 index = 0; index < kProtectedReservationCount; ++index) {
      output->protected_destinations[index] = destinations[index];
    }
    output->protected_graph_failed = graph_failed;
    output->protected_speculative_stale = speculative_stale;
    output->protected_speculative_wasted_bytes =
      speculative_wasted_bytes;
    output->protected_core_prefetch_stale = core_prefetch_stale;
    output->protected_feedback_stale = feedback.stale;
  }
}

__global__ void rob_fast_path_cycle_kernel(
    RobFastPathCycleResult* output) {
  constexpr u32 iterations = 128;
  __shared__ FrontierRobEntry rob[kPersistentFrontierRobCapacity];
  __shared__ u64 preview_handles[kPersistentFrontierRobCapacity];
  __shared__ u16 preview_ranks[kPersistentFrontierRobCapacity];
  __shared__ u64 beam_handles[kPersistentFrontierRobCapacity];
  __shared__ u8 beam_expanded[kPersistentFrontierRobCapacity];
  __shared__ u64 selected_handles[kPersistentFrontierRobCapacity];
  __shared__ u32 selected_ranks[kPersistentFrontierRobCapacity];
  __shared__ u32 commit_slots[kPersistentFrontierRobCapacity];
  __shared__ u64 critical_handles[kPersistentFrontierRobCapacity];
  __shared__ u32 critical_to_commit[kPersistentFrontierRobCapacity];
  __shared__ u32 graph_slots[kPersistentFrontierRobCapacity];
  __shared__ u32 preview_count;
  __shared__ u32 selected_count;
  __shared__ u32 critical_count;
  __shared__ u32 issue_epoch;
  __shared__ u32 physical_issue_span;
  __shared__ u32 speculative_stale;
  __shared__ u64 speculative_wasted_bytes;
  __shared__ u32 issue_epochs;
  __shared__ u64 issue_width_sum;
  __shared__ u64 issue_capacity_sum;
  __shared__ u32 observed_issue_width;
  __shared__ u32 core_stale;
  __shared__ u32 shadow_count;
  __shared__ u32 speculative_promoted;
  __shared__ u32 core_promoted;
  __shared__ u32 critical_hits;
  __shared__ u32 critical_misses;
  __shared__ u32 commit_epochs;
  __shared__ u64 commit_width_sum;
  __shared__ u32 max_commit_width;
  __shared__ u64 prepare_started;
  __shared__ u64 plan_started;
  __shared__ gpu_search::adaptive_frontier::ControllerState controller;
  __shared__ TailFrontierFeedback tail_feedback;

  for (u32 lane = threadIdx.x; lane < kPersistentFrontierRobCapacity;
       lane += blockDim.x) {
    rob[lane] = {};
    preview_handles[lane] = 100 + lane;
    preview_ranks[lane] = static_cast<u16>(lane);
    beam_handles[lane] = 100 + lane;
    beam_expanded[lane] = 0;
  }
  if (threadIdx.x == 0) {
    preview_count = kPersistentFrontierRobCapacity;
    selected_count = 0;
    critical_count = 0;
    issue_epoch = 0;
    physical_issue_span = 0;
    speculative_stale = 0;
    speculative_wasted_bytes = 0;
    issue_epochs = 0;
    issue_width_sum = 0;
    issue_capacity_sum = 0;
    observed_issue_width = 0;
    core_stale = 0;
    shadow_count = 0;
    speculative_promoted = 0;
    core_promoted = 0;
    critical_hits = 0;
    critical_misses = 0;
    commit_epochs = 0;
    commit_width_sum = 0;
    max_commit_width = 0;
    controller =
      gpu_search::adaptive_frontier::make_controller_state(16, 32);
    tail_feedback = {};
    *output = {};
  }
  __syncthreads();

  for (u32 iteration = 0; iteration < iterations; ++iteration) {
    if (threadIdx.x == 0) {
      // Hold the synthetic cycle microbenchmark at the no-tail ASFE fast
      // state. The functional kernel above separately verifies adaptive
      // growth and retained-tail reconciliation.
      controller.current_issue_width = controller.commit_width;
      tail_feedback = {};
      preview_count = kPersistentFrontierRobCapacity;
      prepare_started = clock64();
    }
    __syncthreads();
    prepare_issue_frontier_entries(
      preview_handles, preview_ranks, preview_count, rob, issue_epoch,
      controller, 16, tail_feedback, speculative_stale,
      speculative_wasted_bytes, core_stale, issue_epochs,
      issue_width_sum, issue_capacity_sum, observed_issue_width,
      commit_slots, physical_issue_span);
    if (threadIdx.x == 0) {
      output->prepare_cycles += clock64() - prepare_started;
    }
    for (u32 lane = threadIdx.x; lane < 16; lane += blockDim.x) {
      if (rob[lane].state !=
          static_cast<u8>(FrontierRequestState::issued)) {
        atomicExch(&output->failed, 1u);
      }
      rob[lane].state = static_cast<u8>(FrontierRequestState::validated);
      rob[lane].transfer_bytes = 64;
    }
    __syncthreads();
    if (threadIdx.x == 0) plan_started = clock64();
    __syncthreads();
    plan_commit_frontier(
      beam_handles, beam_expanded, 16, 16, rob, controller, true, 0, 0,
      selected_handles, selected_ranks, commit_slots, selected_count,
      critical_handles, critical_to_commit, critical_count, graph_slots,
      shadow_count, speculative_stale, speculative_wasted_bytes,
      speculative_promoted, core_stale, core_promoted, critical_hits,
      critical_misses, commit_epochs, commit_width_sum, max_commit_width,
      tail_feedback);
    if (threadIdx.x == 0) {
      output->plan_cycles += clock64() - plan_started;
      if (selected_count != 16 || critical_count != 0) {
        output->failed = 2;
      }
    }
    for (u32 lane = threadIdx.x; lane < 16; lane += blockDim.x) {
      if (rob[lane].state ==
          static_cast<u8>(FrontierRequestState::committed)) {
        rob[lane] = {};
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) output->iterations = iterations;
}

__global__ void underhint_force_full_mapping_kernel(
    UnderhintForceFullResult* output) {
  __shared__ FrontierRobEntry rob[kPersistentFrontierRobCapacity];
  __shared__ u64 selected_handles[kPersistentFrontierRobCapacity];
  __shared__ u8 selected_force_full[kPersistentFrontierRobCapacity];
  __shared__ u32 certified_rob_slots[kPersistentFrontierRobCapacity];
  __shared__ u32 critical_to_commit[kPersistentFrontierRobCapacity];
  __shared__ u8 critical_force_full[kPersistentFrontierRobCapacity];
  __shared__ u32 any_force_full;

  const u32 lane = threadIdx.x;
  if (lane < kPersistentFrontierRobCapacity) {
    rob[lane] = {};
    selected_handles[lane] = kInvalidDeviceHandle;
    selected_force_full[lane] = 0xffu;
    certified_rob_slots[lane] = UINT32_MAX;
    critical_to_commit[lane] = UINT32_MAX;
    critical_force_full[lane] = 0xffu;
  }
  if (lane == 0) {
    *output = {};
    selected_handles[0] =
      (u64{7} << gpu_search::kRemoteIncarnationShift) | 101u;
    selected_handles[1] =
      (u64{8} << gpu_search::kRemoteIncarnationShift) | 102u;
    selected_handles[2] =
      (u64{9} << gpu_search::kRemoteIncarnationShift) | 103u;
    selected_handles[3] =
      (u64{10} << gpu_search::kRemoteIncarnationShift) | 104u;

    rob[0].node_handle = selected_handles[0];
    rob[0].state = static_cast<u8>(FrontierRequestState::stale);
    rob[0].validation =
      static_cast<u8>(FrontierValidationState::extent_underhint);

    rob[1].node_handle = selected_handles[1];
    rob[1].state = static_cast<u8>(FrontierRequestState::stale);
    rob[1].validation =
      static_cast<u8>(FrontierValidationState::invalid_snapshot);

    rob[2].node_handle = selected_handles[2];
    rob[2].state = static_cast<u8>(FrontierRequestState::stale);
    rob[2].validation =
      static_cast<u8>(FrontierValidationState::stale_incarnation);

    // Same physical low bits, but a different incarnation: never evidence for
    // selected_handles[2], even on the associative or certified lookup path.
    rob[23].node_handle =
      (u64{11} << gpu_search::kRemoteIncarnationShift) | 103u;
    rob[23].state = static_cast<u8>(FrontierRequestState::stale);
    rob[23].validation =
      static_cast<u8>(FrontierValidationState::extent_underhint);

    certified_rob_slots[0] = 0;
    certified_rob_slots[1] = 1;
    certified_rob_slots[2] = 23;
    certified_rob_slots[3] = 31;
  }
  __syncthreads();

  identify_selected_underhint_force_full(
    selected_handles, 4, rob, certified_rob_slots,
    UnderhintLookupMode::positional, selected_force_full,
    &any_force_full);
  if (lane == 0) output->any_with_underhint = any_force_full;
  if (lane < 4) {
    output->selected_force_full[lane] = selected_force_full[lane];
  }

  // Positional core compaction preserves commit order.
  if (lane < 4) critical_to_commit[lane] = lane;
  __syncthreads();
  remap_critical_underhint_force_full(
    selected_force_full, 4, critical_to_commit, 4,
    critical_force_full);
  if (lane < 4) {
    output->positional_force_full[lane] = critical_force_full[lane];
  }

  // General ROB reconciliation may compact misses in a different order.
  identify_selected_underhint_force_full(
    selected_handles, 4, rob, certified_rob_slots,
    UnderhintLookupMode::associative, selected_force_full,
    &any_force_full);
  if (lane == 0) {
    critical_to_commit[0] = 2;
    critical_to_commit[1] = 0;
    critical_to_commit[2] = 3;
    critical_to_commit[3] = 1;
  }
  __syncthreads();
  remap_critical_underhint_force_full(
    selected_force_full, 4, critical_to_commit, 4,
    critical_force_full);
  if (lane < 4) {
    output->general_force_full[lane] = critical_force_full[lane];
  }

  // A reusable certificate supplies the same position map through its own
  // reconciliation path; the common post-compaction remap remains exact.
  identify_selected_underhint_force_full(
    selected_handles, 4, rob, certified_rob_slots,
    UnderhintLookupMode::certified, selected_force_full,
    &any_force_full);
  if (lane == 0) {
    critical_to_commit[0] = 1;
    critical_to_commit[1] = 2;
    critical_to_commit[2] = 3;
    critical_to_commit[3] = 0;
  }
  __syncthreads();
  remap_critical_underhint_force_full(
    selected_force_full, 4, critical_to_commit, 4,
    critical_force_full);
  if (lane < 4) {
    output->certified_force_full[lane] = critical_force_full[lane];
  }
  if (lane == 0) {
    rob[0].validation =
      static_cast<u8>(FrontierValidationState::invalid_snapshot);
    rob[23].validation =
      static_cast<u8>(FrontierValidationState::invalid_snapshot);
  }
  __syncthreads();
  identify_selected_underhint_force_full(
    selected_handles, 4, rob, certified_rob_slots,
    UnderhintLookupMode::associative, selected_force_full,
    &any_force_full);
  if (lane == 0) output->any_without_underhint = any_force_full;
}

__global__ void owner_validation_kernel(
    PersistentKernelParams params, DirectBatchDescriptor descriptor,
    u32 memory_node) {
  validate_frontier_owner_batch(
    params, descriptor, memory_node, threadIdx.x & 31u);
}

__global__ void dynamic_unknown_validation_kernel(
    PersistentKernelParams params,
    FrontierRobEntry entry,
    u32* output) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  __shared__ DynamicGraphTelemetry telemetry;
  telemetry = {};
  output[0] = static_cast<u32>(validate_frontier_record_local(
    params, 0, entry, &telemetry));
  output[1] = load_dynamic_graph_extent_class(params, entry.node_handle);
  output[2] = telemetry.hint_demotions;
}

void store_u32(u8* destination, u32 value) {
  destination[0] = static_cast<u8>(value);
  destination[1] = static_cast<u8>(value >> 8);
  destination[2] = static_cast<u8>(value >> 16);
  destination[3] = static_cast<u8>(value >> 24);
}

void finalize_graph_record(u8* record) {
  const u16 checksum =
    gpu_search::graph_record_validation::checksum16(record, 32);
  record[2] = static_cast<u8>(checksum);
  record[3] = static_cast<u8>(checksum >> 8);
}

bool certified_reconcile_matches(
    const CertifiedReconcileResult& result, bool issue_wave) {
  constexpr u32 expected_commit_slots[kCertifiedSelectedCount]{
    23, 4, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
  constexpr u32 expected_graph_slots[kCertifiedSelectedCount]{
    kGraphScratchBit | 9u,
    kGraphScratchBit | 27u,
    UINT32_MAX,
    UINT32_MAX,
    UINT32_MAX,
    UINT32_MAX,
  };
  constexpr u64 expected_fetch_handles[kCertifiedMissCount]{
    5002, 5003, 5004, 5005};
  constexpr u32 expected_fetch_positions[kCertifiedMissCount]{2, 3, 4, 5};
  constexpr u8 expected_states[5]{
    static_cast<u8>(FrontierRequestState::committed),
    static_cast<u8>(FrontierRequestState::committed),
    static_cast<u8>(FrontierRequestState::inflight),
    static_cast<u8>(FrontierRequestState::validated),
    static_cast<u8>(FrontierRequestState::validated),
  };
  if (result.failed != 0 ||
      result.critical_fetch_count != kCertifiedMissCount ||
      result.critical_rob_hits != 12 ||
      result.critical_misses != 24 ||
      result.speculative_promoted != 31 ||
      result.core_prefetch_promoted != 41 ||
      result.core_prefetch_stale != 51 ||
      result.feedback_promoted != 61 ||
      result.feedback_core_hits != (issue_wave ? 72u : 70u) ||
      result.feedback_core_misses != (issue_wave ? 84u : 80u) ||
      result.feedback_retained != 61 ||
      result.feedback_stale != 62 ||
      result.feedback_queue_rejects != 63) {
    return false;
  }
  for (u32 position = 0; position < kCertifiedSelectedCount; ++position) {
    if (result.commit_rob_slots[position] !=
          expected_commit_slots[position] ||
        result.graph_record_slots[position] !=
          expected_graph_slots[position]) {
      return false;
    }
  }
  for (u32 index = 0; index < kCertifiedMissCount; ++index) {
    if (result.critical_fetch_handles[index] !=
          expected_fetch_handles[index] ||
        result.critical_fetch_to_commit[index] !=
          expected_fetch_positions[index]) {
      return false;
    }
  }
  for (u32 index = 0; index < 5; ++index) {
    if (result.mapped_states[index] != expected_states[index]) return false;
  }
  return true;
}

bool run_owner_validation_test() {
  OwnerValidationFixture* fixture = nullptr;
  check_cuda(cudaMallocManaged(
               reinterpret_cast<void**>(&fixture),
               sizeof(OwnerValidationFixture)),
             "cudaMallocManaged owner validation");
  std::memset(fixture, 0, sizeof(*fixture));

  constexpr u32 memory_node = 2;
  constexpr u64 static_handle =
    (u64{memory_node} << gpu_search::kRemoteOffsetUnitBits) | 1u;
  constexpr u64 dynamic_handle =
    (u64{7} << gpu_search::kRemoteIncarnationShift) |
    (u64{memory_node} << gpu_search::kRemoteOffsetUnitBits) | 2u;

  // A checksum-valid short Live-Extent record with a canonical zero suffix.
  fixture->records[0][0] = 1;
  store_u32(fixture->records[0] + 8, 0);
  finalize_graph_record(fixture->records[0]);

  // A valid snapshot of an older dynamic incarnation.
  fixture->records[1][0] = 1;
  store_u32(fixture->records[1] + 8, 6);
  finalize_graph_record(fixture->records[1]);

  // Structurally valid but checksum-corrupt.
  fixture->records[2][0] = 1;
  store_u32(fixture->records[2] + 8, 0);
  finalize_graph_record(fixture->records[2]);
  fixture->records[2][16] ^= 1u;

  // A readable header whose two neighbors do not fit in the transferred
  // 16-byte prefix. It may request a query-local full retry, but it is not a
  // checksum-authoritative snapshot and cannot repair a global hint.
  fixture->records[3][0] = 2;
  store_u32(fixture->records[3] + 8, 0);
  finalize_graph_record(fixture->records[3]);

  for (u32 index = 0; index < kPersistentFrontierRobCapacity; ++index) {
    fixture->shards[index] = memory_node;
    fixture->unrelated_shards[index] = memory_node;
    fixture->states[index] =
      static_cast<u8>(FrontierValidationState::unknown);
  }
  fixture->local_iovas[0] = 0;
  fixture->local_iovas[1] = 32;
  fixture->local_iovas[2] = 64;
  fixture->local_iovas[3] = 96;
  fixture->handles[0] = static_handle;
  fixture->handles[1] = dynamic_handle;
  fixture->handles[2] = static_handle;
  fixture->handles[3] = static_handle;
  fixture->bytes[0] = 24;
  fixture->bytes[1] = 32;
  fixture->bytes[2] = 32;
  fixture->bytes[3] = 16;

  PersistentKernelParams params{};
  params.query_slots = 1;
  params.graph_degree = 2;
  params.graph_entry_capacity = 2;
  params.graph_entry_bytes = 32;
  params.direct_local_iova_base =
    reinterpret_cast<u64>(fixture->records);
  params.speculative_graph_request_shards = fixture->shards;
  params.speculative_graph_request_offsets = fixture->offsets;
  params.speculative_graph_request_local_iovas = fixture->local_iovas;
  params.speculative_graph_request_handles = fixture->handles;
  params.speculative_graph_request_bytes = fixture->bytes;
  params.speculative_graph_validation_states = fixture->states;

  DirectBatchDescriptor descriptor{};
  descriptor.request_shards = fixture->shards;
  descriptor.remote_offsets = fixture->offsets;
  descriptor.local_iova_offsets = fixture->local_iovas;
  descriptor.request_bytes = fixture->bytes;
  descriptor.request_count = 4;
  descriptor.bytes = 32;
  owner_validation_kernel<<<1, 32>>>(
    params, descriptor, memory_node);
  check_cuda(cudaGetLastError(), "owner_validation_kernel launch");
  check_cuda(cudaDeviceSynchronize(),
             "owner_validation_kernel synchronize");

  const bool classified =
    fixture->states[0] ==
      static_cast<u8>(FrontierValidationState::valid) &&
    fixture->states[1] ==
      static_cast<u8>(FrontierValidationState::stale_incarnation) &&
    fixture->states[2] ==
      static_cast<u8>(FrontierValidationState::invalid_snapshot) &&
    fixture->states[3] ==
      static_cast<u8>(FrontierValidationState::extent_underhint);

  // Pointer identity is the descriptor type tag. A generic descriptor with
  // look-alike arrays must not write the frontier validation SoA.
  fixture->states[4] =
    static_cast<u8>(FrontierValidationState::unknown);
  descriptor.request_shards = fixture->unrelated_shards;
  descriptor.request_count = 1;
  descriptor.remote_offsets = fixture->offsets + 4;
  descriptor.local_iova_offsets = fixture->local_iovas + 4;
  descriptor.request_bytes = fixture->bytes + 4;
  owner_validation_kernel<<<1, 32>>>(
    params, descriptor, memory_node);
  check_cuda(cudaGetLastError(),
             "owner_validation_identity_kernel launch");
  check_cuda(cudaDeviceSynchronize(),
             "owner_validation_identity_kernel synchronize");
  const bool identity_guard =
    fixture->states[4] ==
      static_cast<u8>(FrontierValidationState::unknown);
  check_cuda(cudaFree(fixture), "cudaFree owner validation");
  return classified && identity_guard;
}

bool run_dynamic_unknown_validation_test() {
  DynamicUnknownValidationFixture* fixture = nullptr;
  check_cuda(cudaMallocManaged(
               reinterpret_cast<void**>(&fixture),
               sizeof(DynamicUnknownValidationFixture)),
             "cudaMallocManaged dynamic unknown validation");
  std::memset(fixture, 0, sizeof(*fixture));

  constexpr u64 dynamic_offset = 0x4000;
  constexpr u32 incarnation = 7;
  fixture->shard = {
    .dynamic_base_offset = dynamic_offset,
    .memory_node = 0,
    .dynamic_record_bytes = 1040,
    .dynamic_hot_offset = 160,
    .dynamic_arena_base_slot = 0,
    .dynamic_arena_slot_count = 1,
  };
  const u64 handle =
    (static_cast<u64>(incarnation) <<
       gpu_search::kRemoteIncarnationShift) |
    (dynamic_offset >> 4);
  FrontierRobEntry entry{};
  entry.node_handle = handle;
  entry.scratch_slot = 0;
  entry.transfer_bytes = 32;

  PersistentKernelParams params{};
  params.shards = &fixture->shard;
  params.num_shards = 1;
  params.graph_degree = 2;
  params.graph_entry_capacity = 2;
  params.graph_entry_bytes = 32;
  params.graph_scratch = fixture->record;
  params.dynamic_graph_extent_enabled = 1;
  params.dynamic_code_arena_states = &fixture->arena_state;
  params.dynamic_code_arena_capacity = 1;

  const auto reset_unknown = [&] {
    fixture->arena_state = gpu_search::make_dynamic_code_tag(
      incarnation, gpu_search::kPersistentDynamicCodeArenaUnknownExtent);
    std::memset(fixture->output, 0, sizeof(fixture->output));
  };
  const auto launch = [&] {
    dynamic_unknown_validation_kernel<<<1, 1>>>(
      params, entry, fixture->output);
    check_cuda(cudaGetLastError(),
               "dynamic_unknown_validation_kernel launch");
    check_cuda(cudaDeviceSynchronize(),
               "dynamic_unknown_validation_kernel synchronize");
  };

  // A real checksum-valid full snapshot is the sole authority allowed to
  // refine a resident same-incarnation UNKNOWN hint. This initialization is
  // deliberately not reported as graph shrinkage.
  fixture->record[0] = 1;
  store_u32(fixture->record + 8, incarnation);
  finalize_graph_record(fixture->record);
  reset_unknown();
  launch();
  const bool valid_refined =
    fixture->output[0] ==
      static_cast<u32>(FrontierValidationState::valid) &&
    fixture->output[1] == 1 && fixture->output[2] == 0;

  // Neither a corrupt snapshot nor a valid snapshot of a recycled
  // incarnation may turn the advisory state into a cache hit for old data.
  reset_unknown();
  fixture->record[16] ^= 1u;
  launch();
  const bool corrupt_retained =
    fixture->output[0] ==
      static_cast<u32>(FrontierValidationState::invalid_snapshot) &&
    fixture->output[1] ==
      gpu_search::kPersistentDynamicCodeArenaUnknownExtent &&
    fixture->arena_state == gpu_search::make_dynamic_code_tag(
      incarnation, gpu_search::kPersistentDynamicCodeArenaUnknownExtent);

  std::memset(fixture->record, 0, sizeof(fixture->record));
  fixture->record[0] = 1;
  store_u32(fixture->record + 8, incarnation + 1);
  finalize_graph_record(fixture->record);
  reset_unknown();
  launch();
  const bool stale_retained =
    fixture->output[0] ==
      static_cast<u32>(FrontierValidationState::stale_incarnation) &&
    fixture->output[1] ==
      gpu_search::kPersistentDynamicCodeArenaUnknownExtent &&
    fixture->arena_state == gpu_search::make_dynamic_code_tag(
      incarnation, gpu_search::kPersistentDynamicCodeArenaUnknownExtent);

  check_cuda(cudaFree(fixture),
             "cudaFree dynamic unknown validation");
  return valid_refined && corrupt_retained && stale_retained;
}

}  // namespace

int main() {
  int devices = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&devices);
  if (device_status != cudaSuccess || devices == 0) {
    std::cout << "SKIP: no CUDA device available\n";
    return 0;
  }
  RobTestResult* device_result = nullptr;
  RobFastPathCycleResult* device_cycles = nullptr;
  EarlyShadowResult* device_early = nullptr;
  LogicalHoleResult* device_logical_hole = nullptr;
  CertifiedReconcileResult* device_certified = nullptr;
  CriticalReservationResult* device_reservation = nullptr;
  UnderhintForceFullResult* device_underhint = nullptr;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_result),
                        2 * sizeof(RobTestResult)), "cudaMalloc");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_cycles),
                        sizeof(RobFastPathCycleResult)), "cudaMalloc cycles");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_early),
                        sizeof(EarlyShadowResult)), "cudaMalloc early");
  check_cuda(cudaMalloc(
               reinterpret_cast<void**>(&device_logical_hole),
               sizeof(LogicalHoleResult)),
             "cudaMalloc logical hole");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_certified),
                        2 * sizeof(CertifiedReconcileResult)),
             "cudaMalloc certified");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_reservation),
                        sizeof(CriticalReservationResult)),
             "cudaMalloc critical reservation");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_underhint),
                        sizeof(UnderhintForceFullResult)),
             "cudaMalloc underhint force-full");
  rob_test_kernel<<<1, 128>>>(device_result, false);
  check_cuda(cudaGetLastError(), "rob_test_kernel launch");
  rob_test_kernel<<<1, 128>>>(device_result + 1, true);
  check_cuda(cudaGetLastError(), "rob_feedback_test_kernel launch");
  rob_fast_path_cycle_kernel<<<1, 128>>>(device_cycles);
  check_cuda(cudaGetLastError(), "rob_fast_path_cycle_kernel launch");
  early_shadow_test_kernel<<<1, 128>>>(device_early);
  check_cuda(cudaGetLastError(), "early_shadow_test_kernel launch");
  logical_hole_test_kernel<<<1, 32>>>(device_logical_hole);
  check_cuda(cudaGetLastError(), "logical_hole_test_kernel launch");
  certified_reconcile_test_kernel<<<1, 32>>>(
    device_certified, true);
  check_cuda(cudaGetLastError(),
             "certified_reconcile_issue_wave_kernel launch");
  certified_reconcile_test_kernel<<<1, 32>>>(
    device_certified + 1, false);
  check_cuda(cudaGetLastError(),
             "certified_reconcile_no_issue_wave_kernel launch");
  critical_reservation_test_kernel<<<1, 32>>>(device_reservation);
  check_cuda(cudaGetLastError(),
             "critical_reservation_test_kernel launch");
  underhint_force_full_mapping_kernel<<<1, 32>>>(device_underhint);
  check_cuda(cudaGetLastError(),
             "underhint_force_full_mapping_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "rob_test_kernel synchronize");
  RobTestResult results[2]{};
  RobFastPathCycleResult cycles{};
  EarlyShadowResult early{};
  LogicalHoleResult logical_hole{};
  CertifiedReconcileResult certified[2]{};
  CriticalReservationResult reservation{};
  UnderhintForceFullResult underhint{};
  check_cuda(cudaMemcpy(results, device_result, sizeof(results),
                        cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  check_cuda(cudaMemcpy(&cycles, device_cycles, sizeof(cycles),
                        cudaMemcpyDeviceToHost), "cudaMemcpy cycles D2H");
  check_cuda(cudaMemcpy(&early, device_early, sizeof(early),
                        cudaMemcpyDeviceToHost), "cudaMemcpy early D2H");
  check_cuda(cudaMemcpy(
               &logical_hole, device_logical_hole,
               sizeof(logical_hole), cudaMemcpyDeviceToHost),
             "cudaMemcpy logical hole D2H");
  check_cuda(cudaMemcpy(certified, device_certified, sizeof(certified),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy certified D2H");
  check_cuda(cudaMemcpy(
               &reservation, device_reservation, sizeof(reservation),
               cudaMemcpyDeviceToHost),
             "cudaMemcpy critical reservation D2H");
  check_cuda(cudaMemcpy(
               &underhint, device_underhint, sizeof(underhint),
               cudaMemcpyDeviceToHost),
             "cudaMemcpy underhint force-full D2H");
  check_cuda(cudaFree(device_result), "cudaFree");
  check_cuda(cudaFree(device_cycles), "cudaFree cycles");
  check_cuda(cudaFree(device_early), "cudaFree early");
  check_cuda(cudaFree(device_logical_hole), "cudaFree logical hole");
  check_cuda(cudaFree(device_certified), "cudaFree certified");
  check_cuda(cudaFree(device_reservation),
             "cudaFree critical reservation");
  check_cuda(cudaFree(device_underhint),
             "cudaFree underhint force-full");
  const RobTestResult& result = results[0];
  const RobTestResult& waste = results[1];
  if (result.failed != 0 || result.first_issue_count != 17 ||
      result.first_issue_span != 17 ||
      result.first_controller_width != 17 ||
      result.first_critical_misses != 0 ||
      result.first_core_promoted != 16 ||
      result.second_issue_count != 17 ||
      result.second_issue_span != 18 ||
      // The surviving speculative record is retention evidence, not a
      // promoted critical hit. Retention alone must not bootstrap another
      // shadow slot, and repeated reconciliation of the same physical record
      // is utility-accounted only once.
      result.second_controller_width != 17 ||
      result.second_critical_misses != 0 ||
      result.second_speculative_promoted != 1 ||
      result.second_controller_promoted != 0 ||
      result.retained_issue_count != 17 ||
      result.retained_issue_span != 18 ||
      result.retained_controller_width != 17 ||
      waste.failed != 0 || waste.first_issue_count != 17 ||
      waste.first_issue_span != 17 ||
      waste.second_issue_count != 17 ||
      waste.second_issue_span != 18 ||
      waste.second_controller_width != 17 ||
      waste.third_issue_count != 17 ||
      waste.third_issue_span != 17 ||
      waste.third_controller_width != 16) {
    std::cerr << "frontier ROB mismatch failed=" << result.failed
              << " first=" << result.first_issue_count << '/'
              << result.first_issue_span << '/'
              << result.first_controller_width << '/'
              << result.first_critical_misses << '/'
              << result.first_core_promoted
              << " second=" << result.second_issue_count << '/'
              << result.second_issue_span << '/'
              << result.second_controller_width << '/'
              << result.second_critical_misses << '/'
              << result.second_speculative_promoted << '/'
              << result.second_controller_promoted << '/'
              << result.retained_issue_count << '/'
              << result.retained_issue_span << '/'
              << result.retained_controller_width
              << " waste=" << waste.failed << '/'
              << waste.first_issue_count << '/'
              << waste.first_issue_span << '/'
              << waste.second_issue_count << '/'
              << waste.second_issue_span << '/'
              << waste.second_controller_width << '/'
              << waste.third_issue_count << '/'
              << waste.third_issue_span << '/'
              << waste.third_controller_width << '\n';
    return 1;
  }
  if (early.failed != 0 || early.first_issue_count != 3 ||
      early.first_issue_epoch != 10 ||
      early.repeat_issue_count != 0 ||
      early.repeat_issue_epoch != 10 ||
      early.first_handle != 1006 || early.last_handle != 1008 ||
      early.first_rank != 6 || early.last_rank != 8) {
    std::cerr << "early shadow mismatch failed=" << early.failed
              << " first=" << early.first_issue_count << '/'
              << early.first_issue_epoch << '/'
              << early.first_handle << '/' << early.first_rank
              << " last=" << early.last_handle << '/'
              << early.last_rank
              << " repeat=" << early.repeat_issue_count << '/'
              << early.repeat_issue_epoch << '\n';
    return 1;
  }
  if (logical_hole.failed != 0 ||
      logical_hole.issue_count != 18 ||
      logical_hole.physical_issue_span !=
        kPersistentFrontierRobCapacity ||
      logical_hole.issue_epochs != 1 ||
      logical_hole.admitted_issue_width_sum != 17 ||
      logical_hole.issue_width_capacity_sum != 18 ||
      logical_hole.observed_admitted_width != 17 ||
      logical_hole.hole_mapping != UINT32_MAX ||
      logical_hole.retained_mapping !=
        kPersistentFrontierRobCapacity - 1) {
    std::cerr << "logical-hole ROB mismatch failed="
              << logical_hole.failed
              << " certificate/admitted/span="
              << logical_hole.issue_count << '/'
              << logical_hole.admitted_issue_width_sum << '/'
              << logical_hole.physical_issue_span
              << " epochs/capacity/observed="
              << logical_hole.issue_epochs << '/'
              << logical_hole.issue_width_capacity_sum << '/'
              << logical_hole.observed_admitted_width
              << " mappings=" << logical_hole.hole_mapping << '/'
              << logical_hole.retained_mapping << '\n';
    return 1;
  }
  if (!certified_reconcile_matches(certified[0], true) ||
      !certified_reconcile_matches(certified[1], false)) {
    for (u32 case_index = 0; case_index < 2; ++case_index) {
      const CertifiedReconcileResult& result = certified[case_index];
      std::cerr
        << "certified reconcile mismatch issue_wave="
        << (case_index == 0 ? 1 : 0)
        << " failed=" << result.failed
        << " fetches=" << result.critical_fetch_count
        << " hits/misses=" << result.critical_rob_hits << '/'
        << result.critical_misses
        << " promoted=" << result.speculative_promoted << '/'
        << result.core_prefetch_promoted
        << " core_stale=" << result.core_prefetch_stale
        << " feedback=" << result.feedback_promoted << '/'
        << result.feedback_core_hits << '/'
        << result.feedback_core_misses << '\n';
    }
    return 1;
  }
  constexpr u32 expected_reservations[kCriticalReservationCount]{
    29, 30, 4, 27, 2};
  bool reservation_destinations_match = true;
  for (u32 index = 0; index < kCriticalReservationCount; ++index) {
    reservation_destinations_match &=
      reservation.destinations[index] == expected_reservations[index];
  }
  bool protected_destinations_match = true;
  for (u32 index = 0; index < kProtectedReservationCount; ++index) {
    protected_destinations_match &=
      reservation.protected_destinations[index] == UINT32_MAX;
  }
  if (reservation.failed != 0 ||
      !reservation_destinations_match ||
      reservation.graph_failed != 0 ||
      reservation.speculative_stale != 9 ||
      reservation.speculative_wasted_bytes != 1384 ||
      reservation.core_prefetch_stale != 12 ||
      reservation.feedback_stale != 15 ||
      reservation.feedback_promoted != 17 ||
      reservation.feedback_retained != 19 ||
      !protected_destinations_match ||
      reservation.protected_graph_failed != 5 ||
      reservation.protected_speculative_stale != 7 ||
      reservation.protected_speculative_wasted_bytes != 1000 ||
      reservation.protected_core_prefetch_stale != 11 ||
      reservation.protected_feedback_stale != 13) {
    std::cerr
      << "critical reservation mismatch failed="
      << reservation.failed
      << " destinations=" << reservation.destinations[0] << '/'
      << reservation.destinations[1] << '/'
      << reservation.destinations[2] << '/'
      << reservation.destinations[3] << '/'
      << reservation.destinations[4]
      << " counters=" << reservation.graph_failed << '/'
      << reservation.speculative_stale << '/'
      << reservation.speculative_wasted_bytes << '/'
      << reservation.core_prefetch_stale << '/'
      << reservation.feedback_stale
      << " protected="
      << reservation.protected_destinations[0] << '/'
      << reservation.protected_destinations[1] << '/'
      << reservation.protected_destinations[2] << '/'
      << reservation.protected_graph_failed << '/'
      << reservation.protected_speculative_stale << '/'
      << reservation.protected_speculative_wasted_bytes << '/'
      << reservation.protected_core_prefetch_stale << '/'
      << reservation.protected_feedback_stale << '\n';
    return 1;
  }
  constexpr u8 expected_selected_force[4]{1, 0, 0, 0};
  constexpr u8 expected_positional_force[4]{1, 0, 0, 0};
  constexpr u8 expected_general_force[4]{0, 1, 0, 0};
  constexpr u8 expected_certified_force[4]{0, 0, 0, 1};
  if (underhint.any_with_underhint != 1 ||
      underhint.any_without_underhint != 0) {
    std::cerr << "underhint any-gate mismatch with/without="
              << underhint.any_with_underhint << '/'
              << underhint.any_without_underhint << '\n';
    return 1;
  }
  for (u32 index = 0; index < 4; ++index) {
    if (underhint.selected_force_full[index] !=
          expected_selected_force[index] ||
        underhint.positional_force_full[index] !=
          expected_positional_force[index] ||
        underhint.general_force_full[index] !=
          expected_general_force[index] ||
        underhint.certified_force_full[index] !=
          expected_certified_force[index]) {
      std::cerr << "underhint force-full mapping mismatch index=" << index
                << " selected/positional/general/certified="
                << static_cast<u32>(underhint.selected_force_full[index])
                << '/'
                << static_cast<u32>(underhint.positional_force_full[index])
                << '/'
                << static_cast<u32>(underhint.general_force_full[index])
                << '/'
                << static_cast<u32>(underhint.certified_force_full[index])
                << '\n';
      return 1;
    }
  }
  if (cycles.failed != 0 || cycles.iterations == 0) {
    std::cerr << "frontier ROB fast-path cycle test failed="
              << cycles.failed << '\n';
    return 1;
  }
  if (!run_owner_validation_test()) {
    std::cerr << "frontier owner validation mismatch\n";
    return 1;
  }
  if (!run_dynamic_unknown_validation_test()) {
    std::cerr << "dynamic UNKNOWN validation/refinement mismatch\n";
    return 1;
  }
  std::cout << "rob_fast_prepare_cycles="
            << cycles.prepare_cycles / cycles.iterations
            << ",rob_fast_plan_cycles="
            << cycles.plan_cycles / cycles.iterations << '\n';
  return 0;
}
