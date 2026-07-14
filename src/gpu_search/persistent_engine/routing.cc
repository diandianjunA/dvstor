#include "gpu_search/persistent_engine/impl.hh"
#include "gpu_search/persistent_engine/cuda_helpers.hh"

namespace gpu_search {

using namespace persistent_engine_detail;
void PersistentSearchEngine::Impl::decode_mutation_payload(const DeltaMutation& mutation,
                             std::vector<f32>& decoded) const {
  std::fill(decoded.begin(), decoded.end(), 0.0f);
  if (mutation.kind == service::storage_owner::MutationKind::erase) return;
  if (mutation.vector.size() == static_cast<size_t>(config.dim) * sizeof(f32)) {
    std::memcpy(decoded.data(), mutation.vector.data(), mutation.vector.size());
  } else if (mutation.vector.size() ==
             vector_dtype_bytes(config.resolved_vector_dtype(), config.dim)) {
    decode_storage_vector_to_float(mutation.vector.data(), config.resolved_vector_dtype(),
                                   config.dim, decoded.data());
  } else {
    throw std::invalid_argument("GPU delta mutation vector has an invalid size");
  }
}

u32 PersistentSearchEngine::Impl::nearest_anchor(const std::vector<f32>& vector, u64 remote_node) const {
  if (anchor_table.count() == 0) return 0;
  const u32 shard = static_cast<u32>(remote_node >> 48);
  if (shard >= index.shards.size()) return 0;
  const u32 begin = anchor_table.shard_offsets[shard];
  const u32 end = anchor_table.shard_offsets[shard + 1];
  if (begin == end) return 0;
  u32 best = begin;
  f32 best_distance = std::numeric_limits<f32>::max();
  for (u32 anchor = begin; anchor < end; ++anchor) {
    f32 distance = 0.0f;
    const f32* candidate = anchor_table.vectors.data() +
      static_cast<size_t>(anchor) * config.dim;
    for (u32 dimension = 0; dimension < config.dim; ++dimension) {
      const f32 difference = vector[dimension] - candidate[dimension];
      distance += difference * difference;
    }
    if (distance < best_distance) {
      best_distance = distance;
      best = anchor;
    }
  }
  return best;
}

u64 PersistentSearchEngine::Impl::graph_cache_key(u64 raw) const {
  const u32 shard = static_cast<u32>(raw >> 48);
  const u64 node_offset = (raw << 16) >> 16;
  if (raw == 0 || shard >= index.shards.size()) {
    throw std::runtime_error("storage returned an invalid GPU graph-cache invalidation");
  }
  const format::ShardRegion& region = index.shards[shard];
  u64 graph_offset = 0;
  if (node_offset >= region.node_base_offset && region.node_stride != 0) {
    const u64 relative = node_offset - region.node_base_offset;
    if (relative % region.node_stride == 0 &&
        relative / region.node_stride < region.node_count) {
      graph_offset = region.graph_base_offset +
        (relative / region.node_stride) * index.layout.graph_entry_bytes;
    }
  }
  if (graph_offset == 0) {
    if (node_offset < region.dynamic_base_offset || region.dynamic_record_bytes == 0 ||
        (node_offset - region.dynamic_base_offset) % region.dynamic_record_bytes != 0) {
      throw std::runtime_error("storage returned a misaligned GPU graph-cache invalidation");
    }
    graph_offset = node_offset + region.dynamic_hot_offset;
  }
  return (static_cast<u64>(shard) << 48) | graph_offset;
}

std::vector<u64> PersistentSearchEngine::Impl::graph_cache_keys(std::span<const u64> raw_nodes) const {
  std::vector<u64> keys;
  keys.reserve(raw_nodes.size());
  for (u64 raw : raw_nodes) keys.push_back(graph_cache_key(raw));
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  if (keys.size() > graph_invalidation_capacity) {
    throw std::runtime_error("GPU navigation graph invalidation batch exceeds capacity");
  }
  return keys;
}

void PersistentSearchEngine::Impl::refresh_anchor_graph_records(std::span<const u64> invalidation_keys) {
  if (invalidation_keys.empty() || anchor_graph_keys_host.empty()) return;
  std::vector<u32> route_slots;
  route_slots.reserve(invalidation_keys.size());
  for (u64 key : invalidation_keys) {
    const auto iterator = std::lower_bound(
      anchor_graph_keys_host.begin(), anchor_graph_keys_host.end(), key);
    if (iterator != anchor_graph_keys_host.end() && *iterator == key) {
      route_slots.push_back(static_cast<u32>(
        iterator - anchor_graph_keys_host.begin()));
    }
  }
  if (route_slots.empty()) return;

  const auto timeout = std::chrono::milliseconds(std::clamp<u32>(
    config.storage_owner_rpc_timeout_ms, 1000, 5000));
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    check_cuda(cudaMemcpyAsync(
                 anchor_graph_readers_host, d_anchor_graph_readers,
                 anchor_graph_keys_host.size() * sizeof(u32),
                 cudaMemcpyDeviceToHost, route_refresh_stream),
               "cudaMemcpyAsync(anchor route readers)");
    check_cuda(cudaStreamSynchronize(route_refresh_stream),
               "cudaStreamSynchronize(anchor route readers)");
    bool busy = false;
    for (u32 slot : route_slots) {
      if (anchor_graph_readers_host[slot] != 0) {
        busy = true;
        break;
      }
    }
    if (!busy) break;
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(
        "anchor route graph refresh timed out waiting for active readers");
    }
    std::this_thread::yield();
  }

  std::vector<NavigationRead> requests;
  std::vector<i32> statuses(route_slots.size(), -EIO);
  requests.reserve(route_slots.size());
  for (u32 slot : route_slots) {
    const u64 key = anchor_graph_keys_host[slot];
    requests.push_back(NavigationRead{
      .remote_offset = (key << 16) >> 16,
      .destination_address = reinterpret_cast<u64>(
        d_anchor_graph_records +
        static_cast<size_t>(slot) * index.layout.graph_entry_bytes),
      .bytes = index.layout.graph_entry_bytes,
      .memory_node = static_cast<u16>(key >> 48),
    });
  }
  control_bootstrapper->read(requests, statuses);
  for (size_t request = 0; request < statuses.size(); ++request) {
    if (statuses[request] <= 0) {
      throw std::runtime_error(
        "anchor route graph refresh RDMA read failed: slot=" +
        std::to_string(route_slots[request]) + " status=" +
        std::to_string(statuses[request]));
    }
    check_cuda(cudaMemcpyAsync(
                 anchor_graph_validation_host,
                 d_anchor_graph_records +
                   static_cast<size_t>(route_slots[request]) *
                     index.layout.graph_entry_bytes,
                 index.layout.graph_entry_bytes, cudaMemcpyDeviceToHost,
                 route_refresh_stream),
               "cudaMemcpyAsync(anchor route validation)");
    check_cuda(cudaStreamSynchronize(route_refresh_stream),
               "cudaStreamSynchronize(anchor route validation)");
    const u16 expected = vamana::hot_graph::load_u16_le(
      anchor_graph_validation_host + 2);
    const u16 actual = vamana::hot_graph::checksum16(
      anchor_graph_validation_host, index.layout.graph_entry_bytes);
    if (anchor_graph_validation_host[0] > index.layout.graph_degree ||
        expected != actual) {
      throw std::runtime_error(
        "anchor route graph refresh produced an invalid record at slot " +
        std::to_string(route_slots[request]));
    }
  }

  check_cuda(cudaMemcpyAsync(
               d_anchor_graph_states, anchor_graph_ready_states_host.data(),
               anchor_graph_ready_states_host.size() * sizeof(u32),
               cudaMemcpyHostToDevice, route_refresh_stream),
             "cudaMemcpyAsync(anchor route ready states)");
  check_cuda(cudaStreamSynchronize(route_refresh_stream),
             "cudaStreamSynchronize(anchor route ready states)");
  engine.telemetry_.graph_route_refreshes.fetch_add(
    route_slots.size(), std::memory_order_relaxed);
}

}  // namespace gpu_search
