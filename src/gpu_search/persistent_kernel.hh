#pragma once

#include <cstdint>

#include "gpu_search/device_ring.cuh"
#include "gpu_search/types.hh"

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace gpu_search {

inline constexpr u32 kPersistentMaxBeam = 256;
inline constexpr u32 kPersistentMaxExact = 256;
inline constexpr u32 kPersistentMaxCodeBits = 1024;
inline constexpr u32 kPersistentMaxEntryPoints = 512;
inline constexpr u32 kPersistentMaxGraphDegree = 255;
inline constexpr u32 kPersistentMaxPrefetch = 8;
inline constexpr u32 kPersistentMaxAnchorProbes = 64;
inline constexpr u32 kPersistentGraphCacheLineBytes = 512;
inline constexpr u32 kDeltaHandleBit = 0x80000000u;
inline constexpr u32 kDeltaHandleMask = 0x7fffffffu;
inline constexpr u32 kDeltaDeleted = 1u;

struct DeviceShardRegion {
  u64 ordinal_base{};
  u64 node_count{};
  u64 node_base_offset{};
  u64 node_stride{};
  u64 graph_base_offset{};
  u64 dynamic_base_offset{};
  u64 code_remote_offset{};
  u64 code_bytes{};
  u32 memory_node{};
  u32 dynamic_record_bytes{};
  u32 dynamic_hot_offset{};
  u32 reserved{};
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
  u32 signature{};
  u64 epoch{};
  u64 superseded_epoch{};
  u64 remote_node{};
  u32 anchor_bucket{};
  u32 reserved{};
};

struct PersistentKernelParams {
  DeviceRingView<QueryDescriptor> submissions;
  DeviceRingView<CompletionDescriptor> completions;
  DeviceRingView<FetchDescriptor> fetches;
  const DeviceShardRegion* shards{};
  u32 num_shards{};
  const u8* rabitq_entries{};
  const f32* centroid{};
  const u32* entry_points{};
  u32 entry_point_count{};
  u32 num_nodes{};
  u32 medoid_ordinal{};
  u32 dim{};
  u32 code_bits{};
  u32 code_storage_bytes{};
  u32 rabitq_entry_bytes{};
  u32 graph_entry_bytes{};
  u32 graph_degree{};
  u32 graph_shard_bits{};
  u32 node_meta_offset{};
  u32 node_record_bytes{};
  u32 vector_bytes{};
  u32 vector_dtype{};
  u32 beam_width{};
  u32 exact_width{};
  u32 gate_width{};
  u32 gate_max_width{};
  f32 gate_margin{};
  u32 warmup_exact_expansions{};
  u32 audit_period{};
  u32 max_expansions{};
  u32 prefetch_depth{};
  u32 visited_capacity{};
  u32 query_slots{};
  u32 direct_backend{};
  u32 direct_region_count{};
  u32 direct_qps_per_node{};
  u32 direct_local_mkey{};
  u64 direct_local_iova_base{};
  u64 direct_timeout_ns{};
  const DirectRemoteRegion* direct_regions{};
  void* const* direct_qps{};
  u8* direct_dump{};
  u32* direct_disabled{};
  i32* direct_error{};
  const DeviceDeltaRecord* delta_records{};
  const f32* delta_vectors{};
  const u8* delta_rabitq_entries{};
  const u32* delta_next{};
  const u32* delta_bucket_heads{};
  const u32* delta_count{};
  u32 delta_capacity{};
  const u32* base_override_keys{};
  const u64* base_override_epochs{};
  u32 base_override_capacity{};
  const u64* delta_remote_keys{};
  const u32* delta_remote_slots{};
  u32 delta_remote_capacity{};
  const f32* anchor_vectors{};
  u32 anchor_count{};
  u32 delta_anchor_probes{};
  f32* anchor_distances{};
  u32* stop{};
  i32* fetch_status{};
  u32 fetch_status_stride{};
  u8* graph_cache{};
  u64* graph_cache_keys{};
  u64* graph_cache_generations{};
  u64* graph_cache_timestamps{};
  u32* graph_cache_states{};
  u32* graph_cache_readers{};
  u32* graph_cache_victims{};
  const u64* graph_cache_generation{};
  u32 graph_cache_sets{};
  u32 graph_cache_ways{};
  u64 graph_cache_ttl_ns{};
  f32* rotated_queries{};
  f32* query_luts{};
  u32* beam_handles{};
  u32* beam_ids{};
  f32* beam_distances{};
  u8* beam_expanded{};
  u32* visited_hash{};
  u8* exact_records{};
  u8* exact_cache{};
  u32 exact_cache_stride{};
  u32 exact_cache_sets{};
  u32 exact_cache_ways{};
  u32* exact_cache_keys{};
  u32* exact_cache_states{};
  u32* exact_cache_readers{};
  u32* exact_cache_victims{};
  u32* result_ids{};
  f32* result_distances{};
};

void launch_persistent_search(cudaStream_t stream, const PersistentKernelParams& params,
                              u32 blocks, u32 threads);
void launch_publish_delta_count(cudaStream_t stream, u32* count, u32 value);
void launch_supersede_delta_record(cudaStream_t stream, DeviceDeltaRecord* records,
                                   u32 slot, u64 epoch);
void launch_insert_base_override(cudaStream_t stream, u32* keys, u64* epochs,
                                 u32 capacity, u32 ordinal, u64 epoch);
void launch_insert_delta_remote(cudaStream_t stream, u64* keys, u32* slots,
                                u32 capacity, u64 remote_node, u32 slot);
void launch_link_delta_bucket(cudaStream_t stream, u32* bucket_heads, u32* next,
                              u32 bucket, u32 slot);
void launch_invalidate_graph_cache(cudaStream_t stream, const u64* invalidation_keys,
                                   u32 invalidation_count, const u64* cache_keys,
                                   u32* cache_states, const u32* cache_readers,
                                   u32 cache_sets, u32 cache_ways);

}  // namespace gpu_search
