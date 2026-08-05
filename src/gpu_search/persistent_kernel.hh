#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "gpu_search/device_ring.cuh"
#include "gpu_search/types.hh"

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace gpu_search {

inline constexpr u32 kPersistentMaxBeam = 128;
inline constexpr u32 kPersistentMaxExact = 256;
inline constexpr u32 kPersistentMaxSubquantizers = 32;
inline constexpr u32 kPersistentMaxGraphDegree = 128;
// The authoritative commit frontier preserves the legacy maximum batch width.
// The speculative ROB is independently bounded by the same compile-time
// capacity, while graph scratch has a disjoint bank for each role so an
// in-flight shadow read can overlap committed decode/PQ/Beam work.
inline constexpr u32 kPersistentMaxPrefetch = 32;
inline constexpr u32 kPersistentFrontierRobCapacity = 32;
inline constexpr u32 kPersistentGraphScratchSlots = kPersistentMaxPrefetch;
inline constexpr u32 kPersistentStableRunScratch = 4 * kPersistentMaxBeam;
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
  // Optional per-request transfer lengths. A null pointer preserves the
  // uniform `bytes` contract used by exact-vector and dynamic-code reads.
  // The pointed-to device storage must remain live until completion_status is
  // published by the exclusive QP owner.
  const u32* request_bytes{};
  i32* completion_status{};
  u64* completion_timestamp_ns{};
  // A split descriptor is published once on the critical queue. After
  // collecting every visible critical descriptor, the owner may append the
  // suffix to unused SQ credit behind a critical CQ fence. Otherwise it
  // rejects the optional suffix without posting another doorbell. Both
  // outcomes retain independent completion words and priority semantics.
  i32* speculative_completion_status{};
  u64* speculative_completion_timestamp_ns{};
  // Frontier/exact batches are bounded by kPersistentMaxExact and RemotePtr
  // exposes at most kPersistentMaxShards.  Narrow fields keep the descriptor
  // at 80 bytes; owner CTAs retain eight descriptors per warp in shared
  // memory, so the former 88-byte layout materially reduced the query
  // workspace available to the unified persistent kernel.
  u16 request_count{};
  u16 memory_node{};
  u32 bytes{};
  u8 priority{};
  u8 flags{};
  u16 critical_request_count{};
};
static_assert(sizeof(DirectBatchDescriptor) == 80);

enum class DirectBatchPriority : u32 {
  critical = 0,
  speculative = 1,
};

// The split suffix is correctness-critical and must be posted in the same SQ
// train as its prefix. The owner inserts a real mlx5 initiator fence on the
// first suffix READ, so the suffix observes memory only after every prefix
// READ has completed. This is used by exact rerank's second header snapshot;
// unlike an ASFE shadow suffix, it is reserved during critical admission and
// can never be rejected merely because the service boundary has no spare SQ
// credit.
inline constexpr u8 kDirectBatchFlagMandatoryFencedTail = 1u << 0;
// A terminal-cache final train has the same mandatory ordering/completion
// contract, but its suffix may contain additional current-header reads after
// the one-to-one validation trailers:
//
//   [M full records][M matching trailers][H cached-record headers]
//
// Keep this distinct from the original exact-snapshot flag so widening the
// suffix cannot silently weaken the established 2*M descriptor ABI.
inline constexpr u8 kDirectBatchFlagMixedMandatoryFencedTail = 1u << 1;
inline constexpr u8 kDirectBatchKnownFlags =
  kDirectBatchFlagMandatoryFencedTail |
  kDirectBatchFlagMixedMandatoryFencedTail;

// exactify_into_beam normalizes successful fenced trains, successful legacy
// fallbacks, and empty shards to status zero. A remaining non-zero status is
// therefore a final transport failure, not an incarnation/tombstone reject.
// Keep the policy host/device so the query-failure boundary is directly
// unit-testable without a live RDMA peer.
#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr bool exact_snapshot_transport_failed(i32 final_status) {
  return final_status != 0;
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr bool exact_rerank_should_retry_route(
    bool exact_fetch_succeeded, u32 route_attempt) {
  return exact_fetch_succeeded && route_attempt == 0;
}

enum class FrontierRequestState : u8 {
  init = 0,
  issued = 1,
  inflight = 2,
  arrived = 3,
  validated = 4,
  committed = 5,
  stale = 6,
};

enum class FrontierValidationState : u8 {
  unknown = 0,
  valid = 1,
  stale_incarnation = 2,
  invalid_snapshot = 3,
  transport_rejected = 4,
  // A structurally readable short prefix proved that the complete adjacency
  // does not fit. This is query-local retry evidence only; the unverified
  // header must never directly update a global extent hint.
  extent_underhint = 5,
};

inline constexpr u8 kFrontierRobFlagEarlyShadow = 1u << 0;
// Retention is useful controller evidence only on the first exact
// certificate transition of one physical speculative read. Without this bit,
// a long-lived ROB entry is counted again every round and can inflate issue
// width without any additional RDMA benefit.
inline constexpr u8 kFrontierRobFlagUtilityAccounted = 1u << 1;

// Query-CTA-local reorder-buffer entry. Payload remains in coalesced global
// graph scratch; this compact metadata is kept in shared memory and is never
// observed by another CTA. In particular, ISSUED/ARRIVED never changes the
// authoritative Beam, expanded bits, or visited table.
struct FrontierRobEntry {
  u64 node_handle{kInvalidDeviceHandle};
  u32 issue_epoch{};
  u32 transfer_bytes{};
  u16 beam_rank{};
  u8 scratch_slot{};
  u8 state{static_cast<u8>(FrontierRequestState::init)};
  u8 validation{static_cast<u8>(FrontierValidationState::unknown)};
  u8 priority{static_cast<u8>(DirectBatchPriority::speculative)};
  u8 flags{};
};
// query_id is the CTA's QueryDescriptor request_id; request_id is
// issue_epoch*capacity+slot; record location remains in the asynchronously
// owned request-offset SoA. Keeping those wave-invariant/derived fields out of
// every entry cuts shared-memory traffic without weakening the logical ROB
// contract.
static_assert(sizeof(FrontierRobEntry) == 24);

enum class QueryRdmaTraceMode : u32 {
  off = 0,
  sampled = 1,
  full = 2,
};

// Completion is intentionally limited to a software-visible owner priority
// fence. Unsplit submissions expose their final READ (or dump WQE); split
// submissions expose independent critical-prefix and speculative-tail fences.
// Physical per-descriptor, per-parent, and per-WQE completion times remain
// unobservable without changing the transport data path.
struct QueryRdmaTraceEvent {
  u64 request_id{};
  u64 issue_timestamp_ns{};
  u64 wait_phase_start_timestamp_ns{};
  u64 completion_timestamp_ns{};
  u64 batch_process_start_timestamp_ns{};
  u32 route_attempt{};
  u32 search_round{};
  u32 snapshot_attempt{};
  u32 target_shard{};
  u32 parent_count{};
  u32 payload_bytes{};
  u32 minimum_bytes_per_parent{};
  u32 maximum_bytes_per_parent{};
};
static_assert(sizeof(QueryRdmaTraceEvent) == 72);

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
  unsigned long long submitted_wqes{};
  unsigned long long submission_wqe_capacity{};
  unsigned long long critical_batches{};
  unsigned long long speculative_batches{};
};

static_assert(sizeof(DirectOwnerProgress) == 64);

enum class BeamMergePolicy : u32 {
  legacy = 0,
  stable_run = 1,
};

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
  u32 commit_width{};
  u32 issue_width{};
  u32 beam_merge_policy{};
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
  const DeviceRingView<DirectBatchDescriptor>*
    direct_speculative_batch_queues{};
  i32* direct_batch_statuses{};
  u64* direct_batch_completion_timestamps_ns{};
  i32* core_batch_statuses{};
  u64* core_batch_completion_timestamps_ns{};
  i32* tail_batch_statuses{};
  u64* tail_batch_completion_timestamps_ns{};
  u32 direct_batch_queue_count{};
  u32* direct_owner_phases{};
  DirectOwnerProgress* direct_owner_progress{};
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
  // Four eight-edge extent classes per aligned word, indexed by static
  // ordinal. Class zero is a header-only 16-byte record, class n covers n*8
  // neighbor slots, and 0xff means unknown/full-record. The device copy is a
  // monotonic high-water cache: a verified header that outgrows its offline
  // hint atomically promotes only its byte. A null pointer keeps the legacy
  // fixed-size graph READ path exactly intact.
  u32* graph_extent_class_words{};
  // Enables incarnation-scoped graph hints piggybacked by dynamic PQ reads.
  // Kept independent from the static sidecar for base-only/complete
  // Live-Extent ablations; zero leaves dynamic records on full graph reads.
  u32 dynamic_graph_extent_enabled{};
  // Per-query-slot global metadata consumed asynchronously by QP-owner warps.
  // Only the first kPersistentMaxPrefetch entries of each query slot are used
  // by graph fetches.
  u32* graph_request_bytes{};
  // Speculative descriptors remain owned by the QP-owner warp while the query
  // CTA decodes committed records and may reuse the dynamic-PQ request arrays.
  // These dedicated SoA arrays therefore have ROB-slot lifetime and eliminate
  // descriptor-pointer aliasing/ABA with critical graph and dynamic-code reads.
  u32* speculative_graph_request_shards{};
  u64* speculative_graph_request_offsets{};
  u64* speculative_graph_request_local_iovas{};
  u32* speculative_graph_request_bytes{};
  // The QP owner validates prefetched graph snapshots before publishing the
  // completion word. Handles and one-byte results share the ROB-slot lifetime
  // of the request SoA, so query CTAs consume validation without rescanning
  // every record after the overlap window.
  u64* speculative_graph_request_handles{};
  u8* speculative_graph_validation_states{};
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
  // The dynamic PQ arena is indexed directly by physical storage slot.  The
  // low 24 bits name the resident incarnation and bits [30:24] carry its
  // graph-extent hint; bit 31 is BUSY while one CTA replaces the payload.
  // Incarnations are monotonic for a physical slot, preventing a delayed
  // reader (including an old graph-hint repair) from overwriting newer data.
  u32* dynamic_code_arena_states{};
  u8* dynamic_code_arena_records{};
  u64 dynamic_code_arena_capacity{};
  u32* dynamic_code_request_shards{};
  u64* dynamic_code_request_offsets{};
  u64* dynamic_code_request_local_iovas{};
  u32* result_ids{};
  f32* result_distances{};
};

// Storage piggybacks a graph-extent class on the four-byte dynamic PQ
// incarnation tag.  RemotePtr already limits incarnations to 24 bits, so this
// changes neither the dynamic record layout nor the PQ RDMA transfer size.
// Arena states reserve the tag's top bit for BUSY and normalize every unknown
// or unrepresentable remote class to 0x7f.  That value is greater than every
// supported graph extent class and therefore maps to a safe full-record read.
inline constexpr u32 kPersistentDynamicCodeTagIncarnationMask =
  (u32{1} << 24) - 1u;
inline constexpr u32 kPersistentDynamicCodeTagExtentShift = 24;
inline constexpr u32 kPersistentDynamicCodeTagExtentMask = u32{0xff} << 24;
inline constexpr u8 kPersistentDynamicCodeArenaUnknownExtent = 0x7fu;
inline constexpr u32 kPersistentDynamicCodeArenaBusy = u32{1} << 31;
inline constexpr u32 kPersistentDynamicCodeArenaIncarnationMask =
  kPersistentDynamicCodeTagIncarnationMask;
inline constexpr u32 kPersistentDynamicCodeArenaExtentMask =
  u32{kPersistentDynamicCodeArenaUnknownExtent} <<
    kPersistentDynamicCodeTagExtentShift;
static_assert(kRemoteMaxIncarnation <
              kPersistentDynamicCodeArenaIncarnationMask);

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr u32 dynamic_code_tag_incarnation(u32 tag) {
  return tag & kPersistentDynamicCodeTagIncarnationMask;
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr u8 dynamic_code_tag_extent_class(u32 tag) {
  const u32 extent = tag >> kPersistentDynamicCodeTagExtentShift;
  return static_cast<u8>(
    extent < kPersistentDynamicCodeArenaUnknownExtent
      ? extent : kPersistentDynamicCodeArenaUnknownExtent);
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr u32 make_dynamic_code_tag(u32 incarnation,
                                           u8 extent_class) {
  const u32 normalized_extent =
    extent_class < kPersistentDynamicCodeArenaUnknownExtent
      ? extent_class : kPersistentDynamicCodeArenaUnknownExtent;
  return (normalized_extent << kPersistentDynamicCodeTagExtentShift) |
    (incarnation & kPersistentDynamicCodeTagIncarnationMask);
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr bool dynamic_code_arena_state_matches(
    u32 state, u32 incarnation) {
  return incarnation != 0 && incarnation <= kRemoteMaxIncarnation &&
    (state & kPersistentDynamicCodeArenaBusy) == 0 &&
    dynamic_code_tag_incarnation(state) == incarnation;
}

// A cache reader samples the state on both sides of the payload load.  Extent
// promotion/demotion for the same incarnation is harmless because it never
// rewrites the immutable PQ bytes; BUSY or a different incarnation means a
// physical-slot replacement overlapped the read and the score must be
// discarded.
#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr bool dynamic_code_arena_read_stable(
    u32 before, u32 after, u32 incarnation) {
  return dynamic_code_arena_state_matches(before, incarnation) &&
    dynamic_code_arena_state_matches(after, incarnation);
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr bool dynamic_code_arena_can_publish(
    u32 observed_state, u32 desired_incarnation) {
  return desired_incarnation != 0 &&
    desired_incarnation <= kRemoteMaxIncarnation &&
    (observed_state & kPersistentDynamicCodeArenaBusy) == 0 &&
    dynamic_code_tag_incarnation(observed_state) < desired_incarnation;
}

// A successful 0 -> BUSY|tag reservation is the only transition that adds a
// physical arena occupant. Later incarnations replace the resident payload in
// the same slot and must not increase occupancy again.
#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr bool dynamic_code_arena_first_occupancy(
    u32 reserved_from_state) {
  return reserved_from_state == 0;
}

// Produce the same-incarnation, monotonic class transition used by graph
// under-hint repair.  Slot reuse or a BUSY publisher makes the transition a
// no-op, so a delayed query can never attach its old class to a new payload.
#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr bool dynamic_code_arena_promoted_extent_state(
    u32 observed_state,
    u32 desired_incarnation,
    u8 requested_extent_class,
    u32& promoted_state) {
  promoted_state = observed_state;
  if (!dynamic_code_arena_state_matches(
        observed_state, desired_incarnation) ||
      requested_extent_class >= kPersistentDynamicCodeArenaUnknownExtent) {
    return false;
  }
  const u8 observed_extent =
    dynamic_code_tag_extent_class(observed_state);
  if (observed_extent == kPersistentDynamicCodeArenaUnknownExtent ||
      requested_extent_class <= observed_extent) {
    return false;
  }
  promoted_state = make_dynamic_code_tag(
    desired_incarnation, requested_extent_class);
  return true;
}

// UNKNOWN is a conservative full-record sentinel rather than an ordinal
// high-water class. A same-incarnation, checksum-authoritative graph snapshot
// may refine it exactly once without republishing the immutable PQ payload.
// Keeping this transition separate from promotion/demotion preserves their
// telemetry semantics: a planned UNKNOWN full read is not a short->full
// fallback, and learning its exact class is not evidence of graph shrinkage.
#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr bool dynamic_code_arena_refined_unknown_extent_state(
    u32 observed_state,
    u32 desired_incarnation,
    u8 observed_graph_class,
    u32& refined_state) {
  refined_state = observed_state;
  if (!dynamic_code_arena_state_matches(
        observed_state, desired_incarnation) ||
      observed_graph_class >= kPersistentDynamicCodeArenaUnknownExtent ||
      dynamic_code_tag_extent_class(observed_state) !=
        kPersistentDynamicCodeArenaUnknownExtent) {
    return false;
  }
  refined_state = make_dynamic_code_tag(
    desired_incarnation, observed_graph_class);
  return true;
}

// Apply asymmetric hysteresis to graph shrinkage.  A verified snapshot must
// be at least two classes below the cached hint before the cache moves down,
// and the retained one-class guard absorbs an immediate small regrowth.  The
// exact incarnation check gives this pure transition the same slot-reuse
// protection as upward repair.
#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr bool dynamic_code_arena_guarded_demoted_extent_state(
    u32 observed_state,
    u32 desired_incarnation,
    u8 observed_graph_class,
    u32& demoted_state) {
  demoted_state = observed_state;
  if (!dynamic_code_arena_state_matches(
        observed_state, desired_incarnation) ||
      observed_graph_class >= kPersistentDynamicCodeArenaUnknownExtent) {
    return false;
  }
  const u8 cached_extent =
    dynamic_code_tag_extent_class(observed_state);
  if (cached_extent == kPersistentDynamicCodeArenaUnknownExtent ||
      static_cast<u32>(observed_graph_class) + 2u > cached_extent) {
    return false;
  }
  demoted_state = make_dynamic_code_tag(
    desired_incarnation, static_cast<u8>(observed_graph_class + 1u));
  return true;
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
PersistentKernelOccupancy inspect_persistent_search_kernel(
  u32 threads, bool enable_asfe = true);

void launch_persistent_search(cudaStream_t stream,
                              const PersistentKernelParams& params,
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
