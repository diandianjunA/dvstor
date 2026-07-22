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
};

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
  u32 request_count{};
  u32 memory_node{};
  u32 bytes{};
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
  f32* decoded_queries{};
  f32* transformed_queries{};
  f32* query_luts{};
  u64* navigation_candidate_handles{};
  f32* navigation_candidate_distances{};
  u64* visited_hash{};
  u8* exact_records{};
  u8* dynamic_code_records{};
  // Per-query, incarnation-tagged navigation-code cache. Dynamic PQ payloads
  // are immutable after a node incarnation is published; the complete remote
  // handle therefore provides the cache's ABA fence.
  u64* dynamic_code_cache_handles{};
  u8* dynamic_code_cache_records{};
  u32* dynamic_code_request_shards{};
  u64* dynamic_code_request_offsets{};
  u64* dynamic_code_request_local_iovas{};
  u32* result_ids{};
  f32* result_distances{};
};

inline constexpr u32 kPersistentDynamicCodeCacheCapacity = 256;
static_assert((kPersistentDynamicCodeCacheCapacity &
               (kPersistentDynamicCodeCacheCapacity - 1)) == 0);

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
