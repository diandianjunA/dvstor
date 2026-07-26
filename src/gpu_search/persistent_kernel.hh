#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "gpu_search/device_ring.cuh"
#include "gpu_search/types.hh"

#ifdef __CUDACC__
#include <cuda/atomic>
#endif

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace gpu_search {

inline constexpr u32 kPersistentMaxBeam = 128;
inline constexpr u32 kPersistentMaxExact = 256;
inline constexpr u32 kPersistentMaxSubquantizers = 32;
inline constexpr u32 kPersistentMaxGraphDegree = 128;
inline constexpr u32 kPersistentMaxPrefetch = 32;
inline constexpr u32 kPersistentScoreChunk = 16;
inline constexpr u32 kPersistentMaxMergeCandidates = 2048;
// RemotePtr dedicates six bits to the physical shard.  GPU routing and RDMA
// status workspaces cover that complete addressable range so a valid 64-shard
// index never becomes GPU-incompatible at runtime.
inline constexpr u32 kPersistentMaxShards = 64;
inline constexpr u32 kPersistentGraphReadBytes = 2048;
inline constexpr u64 kInvalidDeviceHandle = ~u64{0};
inline constexpr u32 kRemoteOffsetUnitBits = 34;
inline constexpr u32 kRemoteShardBits = 6;
inline constexpr u32 kRemoteIncarnationShift = 40;
inline constexpr u64 kRemoteOffsetUnitMask =
  (u64{1} << kRemoteOffsetUnitBits) - 1;
inline constexpr u64 kRemoteShardMask =
  (u64{1} << kRemoteShardBits) - 1;
inline constexpr u32 kRemoteMaxIncarnation = (u32{1} << 24) - 2;
inline constexpr u32 kNodeHeaderIncarnationShift = 32;

// One scoring chunk merges all neighbors with the current beam. Derive its
// width from the actual graph layout so a high (but supported) R does not
// overflow the fixed GPU top-k workspace or require a dataset-specific knob.
#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr u32 persistent_score_chunk_capacity(
    u32 graph_entry_capacity, u32 traversal_beam_width) {
  if (graph_entry_capacity == 0 ||
      traversal_beam_width >= kPersistentMaxMergeCandidates) {
    return 0;
  }
  const u32 available =
    kPersistentMaxMergeCandidates - traversal_beam_width;
  const u32 by_workspace = available / graph_entry_capacity;
  return by_workspace < kPersistentScoreChunk
    ? by_workspace : kPersistentScoreChunk;
}

struct DeviceShardRegion {
  u64 ordinal_base{};
  u64 node_count{};
  u64 node_base_offset{};
  u64 node_stride{};
  u64 graph_base_offset{};
  u64 dynamic_base_offset{};
  u64 control_remote_offset{};
  u64 code_remote_offset{};
  u64 code_bytes{};
  u32 memory_node{};
  u32 dynamic_record_bytes{};
  u32 dynamic_hot_offset{};
  u32 dynamic_code_offset{};
  // Dense GPU PQ arena mapping for reusable dynamic storage slots.  Every
  // physical slot has exactly one GPU slot; incarnation_state decides whether
  // the payload currently belongs to the requested incarnation.
  u64 dynamic_arena_base_slot{};
  u64 dynamic_arena_slot_count{};
};

// Translate a physical dynamic-node offset to its unique GPU arena slot.
// Keeping this helper host/device makes the storage/GPU layout contract
// directly unit-testable.
#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr bool dynamic_code_arena_slot_from_offset(
    const DeviceShardRegion& region, u64 node_offset,
    u64 arena_capacity, u64& arena_slot) {
  if (region.dynamic_record_bytes == 0 ||
      node_offset < region.dynamic_base_offset) {
    return false;
  }
  const u64 relative = node_offset - region.dynamic_base_offset;
  if (relative % region.dynamic_record_bytes != 0) return false;
  const u64 physical_slot = relative / region.dynamic_record_bytes;
  if (physical_slot >= region.dynamic_arena_slot_count ||
      region.dynamic_arena_base_slot > arena_capacity ||
      physical_slot >= arena_capacity - region.dynamic_arena_base_slot) {
    return false;
  }
  arena_slot = region.dynamic_arena_base_slot + physical_slot;
  return true;
}

struct DirectRemoteRegion {
  u64 address{};
  u32 rkey{};
  u32 reserved{};
};

inline constexpr u32 kCentroidRouteMaxLiveEntries = 4;
inline constexpr u32 kCentroidRouteLive = 1u;

struct DeviceCentroidRouteEntry {
  u64 remote_node{};
  u32 generation{};
  u32 flags{};
};

// One seqlock covers the centroid and all live entries of a shard. Query CTAs
// therefore never rank a centroid from one publication and traverse entry
// handles from another.
struct DeviceCentroidRouteShard {
  u64 sequence{};
  u64 command_id{};
  u64 version{};
  u64 vector_count{};
  u32 live_entry_count{};
  u32 reserved{};
};

struct CentroidRouteUpdate {
  u64 version{};
  u64 vector_count{};
  u32 shard{};
  u32 live_entry_count{};
  std::array<DeviceCentroidRouteEntry,
             kCentroidRouteMaxLiveEntries> entries{};
};

static_assert(sizeof(DeviceCentroidRouteEntry) == 16);
static_assert(sizeof(DeviceCentroidRouteShard) == 40);
static_assert(sizeof(CentroidRouteUpdate) == 88);

struct DirectBatchDescriptor {
  const u32* request_shards{};
  const u64* remote_offsets{};
  const u64* local_iova_offsets{};
  i32* completion_status{};
  u64* completion_timestamp_ns{};
  u32 request_count{};
  u32 memory_node{};
  u32 bytes{};
  u32 reserved{};
};

enum class QueryRdmaTraceMode : u32 {
  off = 0,
  sampled = 1,
  full = 2,
};

// Completion is intentionally shard-batch granular. GPUNetIO requests only a
// CQE for the final READ (or dump WQE), so individual parent/WQE completion
// times are not observable without changing the transport data path.
struct QueryRdmaTraceEvent {
  u64 request_id{};
  u64 issue_timestamp_ns{};
  u64 completion_timestamp_ns{};
  u64 batch_process_start_timestamp_ns{};
  u32 search_round{};
  u32 snapshot_attempt{};
  u32 target_shard{};
  u32 parent_count{};
  u32 bytes_per_parent{};
  u32 reserved{};
};

struct QueryRdmaTraceHeader {
  u64 request_id{};
  u32 event_count{};
  u32 overflow{};
  u32 enabled{};
  u32 reserved{};
};

// One cache-line-isolated device progress record per exclusive QP owner. GPU
// threads update these monotonic counters in local device memory; the host
// watchdog periodically copies the compact array only while queries are
// pending. Keeping this off mapped host memory avoids a PCIe atomic per batch.
struct alignas(64) DirectOwnerProgress {
  unsigned long long announced{};
  unsigned long long dequeued{};
  unsigned long long completed{};
  unsigned long long heartbeat{};
  unsigned long long reserved[4]{};
};

static_assert(sizeof(DirectOwnerProgress) == 64);

enum class QueryExpansionPolicy : u32 {
  fixed = 0,
  feedback_hunger = 1,
};

enum class BeamMergePolicy : u32 {
  legacy = 0,
  stable_run = 1,
};

// Query/owner CTAs share the compact controller in the first cache line.
// Diagnostic counters occupy the second cache line.
struct alignas(64) ExpansionPressureState {
  // [15:0] active, [31:16] credit, [47:32] active peak,
  // [63:48] maximum observed credit.
  unsigned long long control{};
  u32 maximum_credit_tiles{};
  u32 reserved0{};
  unsigned long long reserved_control[6]{};

  unsigned long long hunger_grants{};
  unsigned long long congestion_clears{};
  unsigned long long ring_backpressure_events{};
  unsigned long long sq_defer_events{};
  unsigned long long idle_owner_episodes{};
  unsigned long long reserved_counters[3]{};
};

static_assert(sizeof(ExpansionPressureState) == 128);
static_assert(alignof(ExpansionPressureState) == 64);

// One-shot capacity advertised by one exclusive QP owner.  The high 32 bits
// are an owner epoch and the low 32 bits are unclaimed read-WQE units.  A
// query consumes units with one CAS; an owner busy transition advances the
// epoch and revokes every unclaimed unit.  Unlike the old global credit, a
// unit can therefore be consumed by only one query round and only by work
// routed to this QP.
struct alignas(64) QpExpansionLeaseState {
  unsigned long long control{};
  unsigned long long offers{};
  unsigned long long claims{};
  unsigned long long rejects{};
  unsigned long long returns{};
  unsigned long long revocations{};
  unsigned long long stale_returns{};
  unsigned long long reserved{};
};

static_assert(sizeof(QpExpansionLeaseState) == 64);
static_assert(alignof(QpExpansionLeaseState) == 64);

struct QpExpansionLeaseClaim {
  u32 qp{};
  u32 epoch{};
  u32 wqes{};
};

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr unsigned long long make_qp_expansion_lease_control(
    u32 epoch, u32 available_wqes) {
  return (static_cast<unsigned long long>(epoch) << 32) |
    static_cast<unsigned long long>(available_wqes);
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr u32 qp_expansion_lease_epoch(
    unsigned long long control) {
  return static_cast<u32>(control >> 32);
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr u32 qp_expansion_lease_available(
    unsigned long long control) {
  return static_cast<u32>(control);
}

inline constexpr u32 kExpansionPressureFieldMask = 0xffffu;

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr u32 expansion_pressure_active(unsigned long long control) {
  return static_cast<u32>(control) & kExpansionPressureFieldMask;
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr u32 expansion_pressure_credit(unsigned long long control) {
  return static_cast<u32>(control >> 16) & kExpansionPressureFieldMask;
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr u32 expansion_pressure_active_peak(
    unsigned long long control) {
  return static_cast<u32>(control >> 32) & kExpansionPressureFieldMask;
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr u32 expansion_pressure_credit_peak(
    unsigned long long control) {
  return static_cast<u32>(control >> 48) & kExpansionPressureFieldMask;
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr unsigned long long make_expansion_pressure_control(
    u32 active, u32 credit, u32 active_peak, u32 credit_peak) {
  return static_cast<unsigned long long>(
           active & kExpansionPressureFieldMask) |
    (static_cast<unsigned long long>(
       credit & kExpansionPressureFieldMask) << 16) |
    (static_cast<unsigned long long>(
       active_peak & kExpansionPressureFieldMask) << 32) |
    (static_cast<unsigned long long>(
       credit_peak & kExpansionPressureFieldMask) << 48);
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline bool expansion_owner_idle_episode_transition(
    u32 active_queries, bool queue_empty, bool progress_balanced,
    bool obtained_batch, bool& idle_credit_announced) {
  if (active_queries == 0 || obtained_batch) {
    idle_credit_announced = false;
    return false;
  }
  if (queue_empty && progress_balanced && !idle_credit_announced) {
    idle_credit_announced = true;
    return true;
  }
  return false;
}

#ifdef __CUDACC__
__device__ __forceinline__ unsigned long long expansion_pressure_load(
    const ExpansionPressureState* state) {
  if (state == nullptr) return 0;
  cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> control(
    const_cast<unsigned long long&>(state->control));
  return control.load(cuda::memory_order_relaxed);
}

__device__ __forceinline__ void expansion_pressure_query_enter(
    ExpansionPressureState* state) {
  if (state == nullptr) return;
  unsigned long long observed = expansion_pressure_load(state);
  for (;;) {
    const u32 old_active = expansion_pressure_active(observed);
    const u32 active = min(
      old_active + 1u, kExpansionPressureFieldMask);
    const unsigned long long desired = make_expansion_pressure_control(
      active, old_active == 0 ? 0u : expansion_pressure_credit(observed),
      max(expansion_pressure_active_peak(observed), active),
      expansion_pressure_credit_peak(observed));
    const unsigned long long prior =
      atomicCAS(&state->control, observed, desired);
    if (prior == observed) return;
    observed = prior;
  }
}

__device__ __forceinline__ void expansion_pressure_query_exit(
    ExpansionPressureState* state) {
  if (state == nullptr) return;
  unsigned long long observed = expansion_pressure_load(state);
  for (;;) {
    const u32 old_active = expansion_pressure_active(observed);
    if (old_active == 0) return;
    const u32 active = old_active - 1u;
    const unsigned long long desired = make_expansion_pressure_control(
      active, active == 0 ? 0u : expansion_pressure_credit(observed),
      expansion_pressure_active_peak(observed),
      expansion_pressure_credit_peak(observed));
    const unsigned long long prior =
      atomicCAS(&state->control, observed, desired);
    if (prior == observed) return;
    observed = prior;
  }
}

__device__ __forceinline__ bool expansion_pressure_grant_idle(
    ExpansionPressureState* state) {
  if (state == nullptr || state->maximum_credit_tiles == 0) return false;
  unsigned long long observed = expansion_pressure_load(state);
  for (;;) {
    const u32 active = expansion_pressure_active(observed);
    const u32 credit = expansion_pressure_credit(observed);
    if (active == 0 || credit >= state->maximum_credit_tiles) return false;
    const u32 next_credit = credit + 1u;
    const unsigned long long desired = make_expansion_pressure_control(
      active, next_credit, expansion_pressure_active_peak(observed),
      max(expansion_pressure_credit_peak(observed), next_credit));
    const unsigned long long prior =
      atomicCAS(&state->control, observed, desired);
    if (prior == observed) {
      atomicAdd(&state->hunger_grants, 1ULL);
      return true;
    }
    observed = prior;
  }
}

__device__ __forceinline__ void expansion_pressure_clear_credit(
    ExpansionPressureState* state, bool ring_backpressure, bool sq_defer) {
  if (state == nullptr) return;
  unsigned long long observed = expansion_pressure_load(state);
  bool cleared = false;
  while (expansion_pressure_credit(observed) != 0) {
    const unsigned long long desired = make_expansion_pressure_control(
      expansion_pressure_active(observed), 0,
      expansion_pressure_active_peak(observed),
      expansion_pressure_credit_peak(observed));
    const unsigned long long prior =
      atomicCAS(&state->control, observed, desired);
    if (prior == observed) {
      cleared = true;
      break;
    }
    observed = prior;
  }
  if (cleared) atomicAdd(&state->congestion_clears, 1ULL);
  if (ring_backpressure) {
    atomicAdd(&state->ring_backpressure_events, 1ULL);
  }
  if (sq_defer) atomicAdd(&state->sq_defer_events, 1ULL);
}

__device__ __forceinline__ unsigned long long qp_expansion_lease_load(
    const QpExpansionLeaseState* state) {
  if (state == nullptr) return 0;
  cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> control(
    const_cast<unsigned long long&>(state->control));
  return control.load(cuda::memory_order_relaxed);
}

__device__ __forceinline__ bool qp_expansion_lease_publish(
    QpExpansionLeaseState* state, u32 available_wqes) {
  if (state == nullptr || available_wqes == 0) return false;
  unsigned long long observed = qp_expansion_lease_load(state);
  for (;;) {
    if (qp_expansion_lease_available(observed) != 0) return false;
    const unsigned long long desired = make_qp_expansion_lease_control(
      qp_expansion_lease_epoch(observed) + 1u, available_wqes);
    const unsigned long long prior =
      atomicCAS(&state->control, observed, desired);
    if (prior == observed) {
      atomicAdd(&state->offers, 1ULL);
      return true;
    }
    observed = prior;
  }
}

__device__ __forceinline__ void qp_expansion_lease_revoke(
    QpExpansionLeaseState* state) {
  if (state == nullptr) return;
  unsigned long long observed = qp_expansion_lease_load(state);
  for (;;) {
    const u32 available = qp_expansion_lease_available(observed);
    if (available == 0) return;
    const unsigned long long desired = make_qp_expansion_lease_control(
      qp_expansion_lease_epoch(observed) + 1u, 0u);
    const unsigned long long prior =
      atomicCAS(&state->control, observed, desired);
    if (prior == observed) {
      return;
    }
    observed = prior;
  }
}

__device__ __forceinline__ bool qp_expansion_lease_try_claim(
    QpExpansionLeaseState* states, u32 state_count, u32 qp, u32 wqes,
    QpExpansionLeaseClaim& claim) {
  claim = {};
  if (states == nullptr || qp >= state_count || wqes == 0) return false;
  QpExpansionLeaseState& state = states[qp];
  unsigned long long observed = qp_expansion_lease_load(&state);
  for (;;) {
    const u32 available = qp_expansion_lease_available(observed);
    if (available < wqes) {
      return false;
    }
    const u32 epoch = qp_expansion_lease_epoch(observed);
    const unsigned long long desired = make_qp_expansion_lease_control(
      epoch, available - wqes);
    const unsigned long long prior =
      atomicCAS(&state.control, observed, desired);
    if (prior == observed) {
      claim = {.qp = qp, .epoch = epoch, .wqes = wqes};
      return true;
    }
    observed = prior;
  }
}

__device__ __forceinline__ void qp_expansion_lease_return(
    QpExpansionLeaseState* states, u32 state_count,
    const QpExpansionLeaseClaim& claim) {
  if (states == nullptr || claim.qp >= state_count || claim.wqes == 0) return;
  QpExpansionLeaseState& state = states[claim.qp];
  unsigned long long observed = qp_expansion_lease_load(&state);
  for (;;) {
    if (qp_expansion_lease_epoch(observed) != claim.epoch) {
      return;
    }
    const u32 available = qp_expansion_lease_available(observed);
    const u32 returned = available > UINT32_MAX - claim.wqes
      ? UINT32_MAX : available + claim.wqes;
    const unsigned long long desired = make_qp_expansion_lease_control(
      claim.epoch, returned);
    const unsigned long long prior =
      atomicCAS(&state.control, observed, desired);
    if (prior == observed) {
      return;
    }
    observed = prior;
  }
}
#endif

struct PersistentKernelParams {
  DeviceRingView<QueryDescriptor> submissions;
  DeviceRingView<QueryDescriptor> device_submissions;
  DeviceRingView<CompletionDescriptor> completions;
  DeviceRingView<CentroidRoutePublishDescriptor> route_submissions;
  DeviceRingView<CentroidRoutePublishCompletion> route_completions;
  const DeviceShardRegion* shards{};
  u32 num_shards{};
  const u8* pq_codes{};
  const f32* opq_matrix{};
  const f32* pq_centroids{};
  u32 num_nodes{};
  u32 dim{};
  u32 pq_subquantizers{};
  u32 pq_subvector_dim{};
  u32 pq_code_bytes{};
  u32 dynamic_code_record_bytes{};
  u32 graph_entry_bytes{};
  u32 graph_degree{};
  // Total decodable pointer slots: stable graph_degree plus provisional
  // backlink slots. RobustPrune still owns only graph_degree entries.
  u32 graph_entry_capacity{};
  u32 graph_shard_bits{};
  u32 node_meta_offset{};
  u32 node_record_bytes{};
  u32 node_record_stride{};
  u32 node_vector_offset{};
  u32 node_incarnation_offset{};
  u32 vector_bytes{};
  u32 vector_dtype{};
  u32 traversal_beam_width{};
  u32 final_rerank_width{};
  u32 exact_width{};
  u32 max_expansions{};
  u32 prefetch_depth{};
  u32 query_expansion_policy{};
  u32 beam_merge_policy{};
  u32 efficient_batch_cap{};
  u32 visited_capacity{};
  u32 query_slots{};
  u32 direct_region_count{};
  u32 direct_qps_per_node{};
  u32 direct_local_mkey{};
  u64 direct_local_iova_base{};
  u64 direct_timeout_ns{};
  // Route publication is a local control-plane transaction, not an RDMA CQ
  // operation. Keep its liveness deadline independent from data-path latency.
  u64 route_snapshot_timeout_ns{100'000'000ULL};
  const DirectRemoteRegion* direct_regions{};
  void* const* direct_qps{};
  i32* direct_qp_locks{};
  const DeviceRingView<DirectBatchDescriptor>* direct_batch_queues{};
  i32* direct_batch_statuses{};
  u64* direct_batch_completion_timestamps_ns{};
  u32 direct_batch_queue_count{};
  u32* direct_owner_phases{};
  DirectOwnerProgress* direct_owner_progress{};
  ExpansionPressureState* expansion_pressure{};
  QpExpansionLeaseState* expansion_qp_leases{};
  u32 expansion_qp_lease_count{};
  u8* direct_dump{};
  u32* direct_disabled{};
  i32* direct_error{};
  const CentroidRouteUpdate* centroid_route_updates{};
  const f32* centroid_route_centroid_updates{};
  DeviceCentroidRouteShard* centroid_route_shards{};
  DeviceCentroidRouteEntry* centroid_route_entries{};
  f32* shard_centroids{};
  // Even values denote a complete route table generation. The sole route
  // control CTA makes this odd while publishing any shard subset so query
  // CTAs can validate one ranking snapshot across all shards.
  u64* centroid_route_epoch{};
  u32 centroid_route_shard_capacity{};
  u32 centroid_route_entry_capacity{};
  u32* stop{};
  u32* kernel_ready_count{};
  u32 direct_owner_block_count{};
  u32 query_block_count{};
  u32* query_kernel_ready_count{};
  u32* dispatcher_kernel_ready_count{};
  u32* control_kernel_ready_count{};
  u8* graph_scratch{};
  QueryRdmaTraceHeader* query_rdma_trace_headers{};
  QueryRdmaTraceEvent* query_rdma_trace_events{};
  u32 query_rdma_trace_mode{};
  u32 query_rdma_trace_sample_rate{};
  u32 query_rdma_trace_events_per_query{};
  f32* decoded_queries{};
  f32* transformed_queries{};
  f32* query_luts{};
  u64* navigation_candidate_handles{};
  f32* navigation_candidate_distances{};
  u64* visited_hash{};
  u8* exact_records{};
  u8* dynamic_code_records{};
  // The dynamic PQ arena is indexed directly by physical storage slot.  A
  // state is either 0 (not resident), an incarnation, or BUSY|incarnation
  // while one CTA replaces the payload. Incarnations are monotonic for a
  // physical slot, preventing a delayed reader from overwriting newer data.
  u32* dynamic_code_arena_states{};
  u8* dynamic_code_arena_records{};
  u64 dynamic_code_arena_capacity{};
  u32* dynamic_code_request_shards{};
  u64* dynamic_code_request_offsets{};
  u64* dynamic_code_request_local_iovas{};
  u32* result_ids{};
  f32* result_distances{};
};

inline constexpr u32 kPersistentDynamicCodeArenaBusy = u32{1} << 31;
inline constexpr u32 kPersistentDynamicCodeArenaIncarnationMask =
  kPersistentDynamicCodeArenaBusy - 1;
static_assert(kRemoteMaxIncarnation <
              kPersistentDynamicCodeArenaIncarnationMask);

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr bool dynamic_code_arena_can_publish(
    u32 observed_state, u32 desired_incarnation) {
  return desired_incarnation != 0 &&
    desired_incarnation <= kRemoteMaxIncarnation &&
    (observed_state & kPersistentDynamicCodeArenaBusy) == 0 &&
    (observed_state & kPersistentDynamicCodeArenaIncarnationMask) <
      desired_incarnation;
}

struct PersistentKernelOccupancy {
  u32 active_blocks_per_sm{};
  u32 registers_per_thread{};
  size_t static_shared_bytes{};
  u32 max_threads_per_block{};
};

// Query the resource footprint of the exact unified kernel binary.  An
// analytical register-only estimate is not sufficient because CUDA also
// accounts for allocation granularity and every other per-CTA resource.
PersistentKernelOccupancy inspect_persistent_search_kernel(u32 threads);

void launch_persistent_search(cudaStream_t stream, const PersistentKernelParams& params,
                              u32 blocks, u32 threads);
void launch_direct_read_owners(cudaStream_t stream,
                               const PersistentKernelParams& params,
                               u32 queue_count, u32 threads);
void launch_gpunetio_owner_read_probe(
  cudaStream_t stream, const PersistentKernelParams& params,
  u32* request_shards, u64* remote_offsets, u64* local_iova_offsets,
  u8* destinations, u32 destination_stride, i32* statuses,
  u32* completed, u32* phases, u32 queue_count);
void launch_gpunetio_locked_read_probe(cudaStream_t stream,
                                       const PersistentKernelParams& params,
                                       u8* destinations, u32 destination_stride,
                                       i32* statuses, u32* completed,
                                       u32 blocks, u32 iterations);
void launch_gpunetio_batched_read_probe(cudaStream_t stream,
                                        const PersistentKernelParams& params,
                                        u8* destinations, u32 destination_stride,
                                        i32* statuses, u32* completed,
                                        u32 blocks, u32 batch_size);
}  // namespace gpu_search
