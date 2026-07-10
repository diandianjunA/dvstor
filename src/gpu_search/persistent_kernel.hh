#pragma once

#include <cstdint>

#include "gpu_search/device_ring.cuh"
#include "gpu_search/types.hh"

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace gpu_search {

inline constexpr u32 kPersistentMaxBeam = 256;
inline constexpr u32 kPersistentMaxExact = 64;
inline constexpr u32 kPersistentMaxCodeBits = 1024;
inline constexpr u32 kPersistentMaxHotDegree = 32;
inline constexpr u32 kPersistentMaxEntryPoints = 512;

struct DeviceNodeRecord {
  u64 remote_node{};
  u64 cold_page_offset{};
  u32 cold_record_offset{};
  u32 generation{1};
  u32 hot_neighbor_begin{};
  u16 hot_neighbor_count{};
  u16 shard{};
  u32 flags{};
};

struct DirectRemoteRegion {
  u64 address{};
  u32 rkey{};
  u32 reserved{};
};

inline constexpr u32 kDeltaDeleted = 1u;

struct DeviceDeltaRecord {
  u32 id{};
  u32 generation{};
  u32 flags{};
  u32 signature{};
  u64 epoch{};
  u64 superseded_epoch{};
};

struct PersistentKernelParams {
  DeviceRingView<QueryDescriptor> submissions;
  DeviceRingView<CompletionDescriptor> completions;
  DeviceRingView<FetchDescriptor> fetches;
  const DeviceNodeRecord* nodes{};
  const u32* hot_neighbors{};
  const u8* rabitq_entries{};
  const f32* centroid{};
  const u32* entry_points{};
  u32 entry_point_count{};
  u32 num_nodes{};
  u32 medoid_id{};
  u32 dim{};
  u32 code_bits{};
  u32 code_storage_bytes{};
  u32 rabitq_entry_bytes{};
  u32 vector_offset{};
  u32 vector_bytes{};
  u32 vector_dtype{};
  u32 beam_width{};
  u32 exact_width{};
  u32 max_expansions{};
  u32 cold_expansions{};
  u32 visited_capacity{};
  u32 query_slots{};
  u32 graph_page_bytes{};
  u32 id_encoding_bytes{};
  u32 direct_backend{};
  u32 direct_region_count{};
  u32 direct_qps_per_node{};
  u32 direct_local_mkey{};
  u64 direct_local_iova_base{};
  const DirectRemoteRegion* direct_regions{};
  void* const* direct_qps{};
  u8* direct_dump{};
  const DeviceDeltaRecord* delta_records{};
  const f32* delta_vectors{};
  const u8* delta_rabitq_entries{};
  const u64* base_override_epochs{};
  const u32* delta_count{};
  u32 delta_capacity{};
  u32* stop{};
  i32* fetch_status{};
  u32 fetch_status_stride{};
  u8* graph_page_cache{};
  u64* graph_page_cache_keys{};
  u32* graph_page_cache_locks{};
  u32 graph_page_cache_slots{};
  f32* rotated_queries{};
  u32* beam_ids{};
  f32* beam_distances{};
  u8* beam_expanded{};
  u32* visited_hash{};
  u8* exact_vectors{};
  u32* result_ids{};
  f32* result_distances{};
};

void launch_persistent_search(cudaStream_t stream, const PersistentKernelParams& params,
                              u32 blocks, u32 threads);

}  // namespace gpu_search
