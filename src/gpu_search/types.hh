#pragma once

#include <atomic>
#include <cstdint>
#include <string_view>

namespace gpu_search {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using f32 = float;

enum class EngineKind : u8 {
  legacy,
  gpu_persistent,
};

enum class RemoteBackendKind : u8 {
  local,
  verbs_proxy,
  gpunetio,
};

inline constexpr std::string_view to_string(EngineKind kind) {
  return kind == EngineKind::gpu_persistent ? "gpu_persistent" : "legacy";
}

inline constexpr std::string_view to_string(RemoteBackendKind kind) {
  switch (kind) {
    case RemoteBackendKind::local: return "local";
    case RemoteBackendKind::verbs_proxy: return "verbs_proxy";
    case RemoteBackendKind::gpunetio: return "gpunetio";
  }
  return "local";
}

inline bool parse_engine_kind(std::string_view value, EngineKind& out) {
  if (value == "legacy") {
    out = EngineKind::legacy;
    return true;
  }
  if (value == "gpu_persistent") {
    out = EngineKind::gpu_persistent;
    return true;
  }
  return false;
}

inline bool parse_remote_backend_kind(std::string_view value, RemoteBackendKind& out) {
  if (value == "local") {
    out = RemoteBackendKind::local;
    return true;
  }
  if (value == "verbs_proxy") {
    out = RemoteBackendKind::verbs_proxy;
    return true;
  }
  if (value == "gpunetio") {
    out = RemoteBackendKind::gpunetio;
    return true;
  }
  return false;
}

struct QueryDescriptor {
  u64 request_id{};
  u64 snapshot_epoch{};
  u64 query_device_address{};
  u64 result_device_address{};
  u32 query_slot{};
  u32 result_capacity{};
  u16 dim{};
  u16 k{};
  u8 query_dtype{};
  u8 flags{};
  u16 reserved{};
};

struct CompletionDescriptor {
  u64 request_id{};
  u64 snapshot_epoch{};
  u64 gpu_cycles{};
  u32 query_slot{};
  u32 result_count{};
  i32 status{};
  u32 remote_pages{};
  u32 exact_vectors{};
  u32 cache_hits{};
};

enum class FetchKind : u8 {
  graph_record,
  node_record,
  code,
};

struct FetchDescriptor {
  u64 request_id{};
  u64 remote_offset{};
  u64 destination_address{};
  u32 bytes{};
  u16 memory_node{};
  u8 kind{};
  u8 flags{};
  u32 destination_lkey{};
  u32 sequence{};
};

struct FetchCompletion {
  u64 request_id{};
  u32 sequence{};
  i32 status{};
  u32 bytes{};
  u32 reserved{};
};

struct TelemetrySnapshot {
  u64 queries_submitted{};
  u64 queries_completed{};
  u64 batches{};
  u64 batch_queries{};
  u64 submission_wait_ns{};
  u64 completion_wait_ns{};
  u64 gpu_active_ns{};
  u64 rdma_read_ops{};
  u64 rdma_read_bytes{};
  u64 rdma_merged_requests{};
  u64 direct_path_failures{};
  u64 graph_page_requests{};
  u64 graph_page_cache_hits{};
  u64 exact_vector_reads{};
  u64 delta_queries{};
  u64 mutations_published{};
  u64 delta_compactions{};
  u64 base_entries_merged{};
  u64 delta_live_entries{};
  u64 visibility_ns_total{};
  u64 visibility_ns_max{};
};

class Telemetry {
public:
  TelemetrySnapshot snapshot() const;
  void reset();

  std::atomic<u64> queries_submitted{0};
  std::atomic<u64> queries_completed{0};
  std::atomic<u64> batches{0};
  std::atomic<u64> batch_queries{0};
  std::atomic<u64> submission_wait_ns{0};
  std::atomic<u64> completion_wait_ns{0};
  std::atomic<u64> gpu_active_ns{0};
  std::atomic<u64> rdma_read_ops{0};
  std::atomic<u64> rdma_read_bytes{0};
  std::atomic<u64> rdma_merged_requests{0};
  std::atomic<u64> direct_path_failures{0};
  std::atomic<u64> graph_page_requests{0};
  std::atomic<u64> graph_page_cache_hits{0};
  std::atomic<u64> exact_vector_reads{0};
  std::atomic<u64> delta_queries{0};
  std::atomic<u64> mutations_published{0};
  std::atomic<u64> delta_compactions{0};
  std::atomic<u64> base_entries_merged{0};
  std::atomic<u64> delta_live_entries{0};
  std::atomic<u64> visibility_ns_total{0};
  std::atomic<u64> visibility_ns_max{0};
};

}  // namespace gpu_search
