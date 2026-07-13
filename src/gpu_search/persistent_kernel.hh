#pragma once

#include <cstdint>

#include "gpu_search/device_ring.cuh"
#include "gpu_search/types.hh"

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace gpu_search {

inline constexpr u32 kPersistentMaxBeam = 128;
inline constexpr u32 kPersistentMaxExact = 256;
inline constexpr u32 kPersistentMaxSubquantizers = 32;
inline constexpr u32 kPersistentMaxEntryPoints = 512;
inline constexpr u32 kPersistentMaxGraphDegree = 128;
inline constexpr u32 kPersistentMaxPrefetch = 32;
inline constexpr u32 kPersistentScoreChunk = 16;
inline constexpr u32 kPersistentMaxMergeCandidates = 2048;
inline constexpr u32 kPersistentMaxShards = 16;
inline constexpr u32 kPersistentMaxAnchorProbes = 64;
inline constexpr u32 kPersistentQueryThreads = 128;
inline constexpr u32 kPersistentGraphCacheLineBytes = 512;
inline constexpr u32 kDeltaHandleBit = 0x80000000u;
inline constexpr u32 kDeltaHandleMask = 0x7fffffffu;
inline constexpr u32 kDeltaDeleted = 1u;
inline constexpr u32 kDeltaDurable = 1u << 1;
inline constexpr u32 kBaseOverrideEmpty = UINT32_MAX;
inline constexpr u32 kBaseOverrideTombstone = UINT32_MAX - 1;
inline constexpr u64 kDeltaRemoteEmpty = 0;
inline constexpr u64 kDeltaRemoteTombstone = UINT64_MAX;

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

struct DeviceDeltaRecord {
  u32 id{};
  u32 generation{};
  u32 flags{};
  u32 base_ordinal{kBaseOverrideEmpty};
  u64 epoch{};
  u64 superseded_epoch{};
  u64 remote_node{};
  u32 anchor_bucket{};
  u32 reserved{};
};

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

struct PersistentKernelParams {
  DeviceRingView<QueryDescriptor> submissions;
  DeviceRingView<CompletionDescriptor> completions;
  DeviceRingView<DeltaPublishDescriptor> delta_submissions;
  DeviceRingView<DeltaPublishCompletion> delta_completions;
  const DeviceShardRegion* shards{};
  u32 num_shards{};
  const u8* pq_codes{};
  const f32* opq_matrix{};
  const f32* pq_centroids{};
  const u32* entry_points{};
  u32 entry_point_count{};
  u32 num_nodes{};
  u32 medoid_ordinal{};
  u32 dim{};
  u32 pq_subquantizers{};
  u32 pq_subvector_dim{};
  u32 pq_code_bytes{};
  u32 graph_entry_bytes{};
  u32 graph_degree{};
  u32 graph_shard_bits{};
  u32 node_meta_offset{};
  u32 node_record_bytes{};
  u32 vector_bytes{};
  u32 vector_dtype{};
  u32 traversal_beam_width{};
  u32 final_rerank_width{};
  u32 entry_seed_count{};
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
  const DirectRemoteRegion* direct_regions{};
  void* const* direct_qps{};
  i32* direct_qp_locks{};
  const DeviceRingView<DirectBatchDescriptor>* direct_batch_queues{};
  i32* direct_batch_statuses{};
  u32 direct_batch_queue_count{};
  u8* direct_dump{};
  u32* direct_disabled{};
  i32* direct_error{};
  DeviceDeltaRecord* delta_records{};
  u8* delta_vectors{};
  u8* delta_pq_codes{};
  const u32* delta_staging_slots{};
  const DeviceDeltaRecord* delta_staging_records{};
  const u8* delta_staging_vectors{};
  f32* delta_encode_scratch{};
  u32* delta_next{};
  u32* delta_prev{};
  u32* delta_remote_positions{};
  u32* delta_bucket_heads{};
  u32* delta_count{};
  u32 delta_capacity{};
  u32* base_override_keys{};
  u64* base_override_epochs{};
  u32 base_override_capacity{};
  u32* permanent_override_bits{};
  u32 permanent_override_words{};
  u64* delta_remote_keys{};
  u32* delta_remote_slots{};
  u32 delta_remote_capacity{};
  const DeltaSupersedeUpdate* delta_supersede_updates{};
  const DeltaOverrideUpdate* delta_override_updates{};
  const DeltaDurableUpdate* delta_durable_updates{};
  const u64* graph_invalidation_keys{};
  const f32* anchor_vectors{};
  const u32* anchor_handles{};
  const u8* anchor_pq_codes{};
  const u64* anchor_graph_keys{};
  const u8* anchor_graph_records{};
  u32* anchor_graph_states{};
  u32* anchor_graph_readers{};
  u32 anchor_graph_count{};
  u32 anchor_count{};
  u32 delta_anchor_probes{};
  u32* stop{};
  u8* graph_cache{};
  u8* graph_scratch{};
  u64* graph_cache_keys{};
  u64* graph_cache_generations{};
  u64* graph_cache_timestamps{};
  u32* graph_cache_states{};
  u32* graph_cache_readers{};
  u32* graph_cache_victims{};
  u64* graph_admission_keys{};
  u32* graph_admission_victims{};
  u32 graph_admission_sets{};
  const u64* graph_cache_generation{};
  u32 graph_cache_sets{};
  u32 graph_cache_ways{};
  u64 graph_cache_ttl_ns{};
  f32* decoded_queries{};
  f32* transformed_queries{};
  f32* query_luts{};
  u32* navigation_candidate_handles{};
  f32* navigation_candidate_distances{};
  u32* visited_hash{};
  u8* exact_records{};
  u8* dynamic_code_records{};
  u32* dynamic_code_request_shards{};
  u64* dynamic_code_request_offsets{};
  u64* dynamic_code_request_local_iovas{};
  u8* exact_cache{};
  u32 exact_cache_stride{};
  u32 exact_cache_sets{};
  u32 exact_cache_ways{};
  u32* exact_cache_keys{};
  u32* exact_cache_states{};
  u32* exact_cache_readers{};
  u32* exact_cache_victims{};
  u32* exact_admission_keys{};
  u32* exact_admission_victims{};
  u32 exact_admission_sets{};
  u32* result_ids{};
  f32* result_distances{};
};

void launch_persistent_search(cudaStream_t stream, const PersistentKernelParams& params,
                              u32 blocks, u32 threads);
void launch_direct_read_owners(cudaStream_t stream,
                               const PersistentKernelParams& params,
                               u32 queue_count, u32 threads);
void launch_gather_anchor_codes(cudaStream_t stream, const u8* base_codes,
                                const u32* anchor_handles, u8* anchor_codes,
                                u32 anchor_count, u32 code_bytes,
                                u32 node_count);
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
