#include "memory_node/memory_node.hh"

#include <algorithm>
#include <cstring>

namespace {

using Configuration = configuration::IndexConfiguration;
using BeamEntry = memory_node_detail::BeamEntry;
using NodeSnapshot = memory_node_detail::NodeSnapshot;
using BeamEntrySerialized = service::storage_owner::BeamEntrySerialized;
using SearchHandoffRequestHeader = service::storage_owner::SearchHandoffRequestHeader;
using SearchHandoffResponseHeader = service::storage_owner::SearchHandoffResponseHeader;
using PeerRpcType = service::storage_owner::PeerRpcType;
using InsertStatus = service::storage_owner::InsertStatus;
static constexpr u32 kPeerRpcMagic = service::storage_owner::kPeerRpcMagic;
using service::storage_owner::handoff_query_vector;
using service::storage_owner::handoff_request_beam;
using service::storage_owner::handoff_request_visited;
using service::storage_owner::handoff_response_beam;
using service::storage_owner::handoff_response_visited;
using service::storage_owner::search_handoff_response_bytes;

inline bool ptr_in_bounds(RemotePtr rptr, u64 shard_cap) {
  if (rptr.is_null()) {
    return false;
  }
  return rptr.byte_offset() + VamanaNode::total_size() <= shard_cap;
}

}  // namespace

// Remote-side search handoff handler.
// Processes an incoming handoff request by doing local-only beam search,
// then sends the response back.
bool MemoryNode::handle_search_handoff_rpc(u32 source_shard,
                                           const SearchHandoffRequestHeader* req,
                                           const byte_t* /*payload*/,
                                           const Configuration& config) {
  const u32 vector_bytes = req->vector_bytes;
  const u32 beam_count = req->rpc.item_count;

  // Unpack query vector
  const auto* query_vec = handoff_query_vector(req);
  const span<const element_t> query{
      reinterpret_cast<const element_t*>(query_vec),
      VamanaNode::DIM};

  // Unpack beam entries
  const auto* serialized_beam = handoff_request_beam(req, vector_bytes);
  vec<BeamEntry> beam;
  beam.reserve(beam_count);
  for (u32 i = 0; i < beam_count; ++i) {
    const RemotePtr rptr{serialized_beam[i].rptr_raw};
    const bool is_local = local_shard(rptr.memory_node());
    beam.push_back({rptr, serialized_beam[i].distance, !is_local});
  }

  // Unpack visited set from the request payload (follows beam entries)
  hashset_t<RemotePtr> visited;
  hashset_t<RemotePtr> request_visited;
  const u32 req_visited_count = req->visited_count;
  const byte_t* req_visited_raws = handoff_request_visited(req, vector_bytes, beam_count);
  visited.reserve(req_visited_count + req->beam_width * config.R);
  request_visited.reserve(req_visited_count);
  for (u32 i = 0; i < req_visited_count; ++i) {
    u64 raw{};
    std::memcpy(&raw, req_visited_raws + static_cast<size_t>(i) * sizeof(raw), sizeof(raw));
    const RemotePtr rptr{raw};
    visited.insert(rptr);
    request_visited.insert(rptr);
  }

  // Phase 1: Compute precise distances for local unexpanded beam entries
  const u32 snapshot_batch = req->snapshot_batch;
  const u64 shard_cap = mn_memory_bytes_;
  vec<RemotePtr> local_unexpanded;
  for (auto& entry : beam) {
    if (!entry.expanded && local_shard(entry.rptr.memory_node()) &&
        ptr_in_bounds(entry.rptr, shard_cap)) {
      local_unexpanded.push_back(entry.rptr);
    }
  }
  for (size_t begin = 0; begin < local_unexpanded.size(); begin += snapshot_batch) {
    const size_t end = std::min(local_unexpanded.size(), begin + snapshot_batch);
    vec<RemotePtr> batch(local_unexpanded.begin() + begin, local_unexpanded.begin() + end);
    vec<NodeSnapshot> snapshots = read_node_snapshots_batched(batch, config);
    for (const NodeSnapshot& snapshot : snapshots) {
      const distance_t dist = distance_to_stored_vector(query, snapshot.vector_data.data(), config);
      // Update distance in beam
      for (auto& entry : beam) {
        if (entry.rptr == snapshot.rptr) {
          entry.distance = dist;
          break;
        }
      }
    }
  }

  // Phase 2: Expand all local nodes
  expand_all_local_nodes(beam, visited, query, config, storage_id_, nullptr);

  // Phase 3: Build and send response
  vec<BeamEntrySerialized> response_beam;
  vec<u64> new_visited_raws;
  response_beam.reserve(beam.size());
  for (const auto& entry : beam) {
    if (!entry.expanded) {
      response_beam.push_back({entry.rptr.raw_address, entry.distance});
    }
  }
  // Collect newly discovered nodes for visited set
  for (const RemotePtr& entry : visited) {
    if (!request_visited.contains(entry)) {
      new_visited_raws.push_back(entry.raw_address);
    }
  }

  const size_t fixed_response_bytes = search_handoff_response_bytes(
    static_cast<u32>(response_beam.size()), 0);
  if (fixed_response_bytes > peer_rpc_runtime_.message_bytes) {
    return false;
  }
  const size_t max_new_visited =
    (peer_rpc_runtime_.message_bytes - fixed_response_bytes) / sizeof(u64);
  if (new_visited_raws.size() > max_new_visited) {
    new_visited_raws.resize(max_new_visited);
  }
  const size_t response_bytes = search_handoff_response_bytes(
    static_cast<u32>(response_beam.size()),
    static_cast<u32>(new_visited_raws.size()));
  vec<byte_t> response_msg(response_bytes);
  auto* resp = reinterpret_cast<SearchHandoffResponseHeader*>(response_msg.data());
  resp->rpc.magic = kPeerRpcMagic;
  resp->rpc.type = static_cast<u32>(PeerRpcType::search_handoff_response);
  resp->rpc.source_shard = storage_id_;
  resp->rpc.item_count = static_cast<u32>(response_beam.size());
  resp->rpc.request_id = req->rpc.request_id;
  resp->rpc.status = static_cast<u32>(InsertStatus::ok);
  resp->rpc.reserved = 0;
  resp->updated_beam_count = static_cast<u32>(response_beam.size());
  resp->new_visited_count = static_cast<u32>(new_visited_raws.size());
  resp->total_visited_count = static_cast<u32>(visited.size());
  resp->reserved = 0;

  auto* resp_beam = handoff_response_beam(resp);
  std::memcpy(resp_beam, response_beam.data(),
              response_beam.size() * sizeof(BeamEntrySerialized));
  auto* resp_visited = handoff_response_visited(resp, static_cast<u32>(response_beam.size()));
  std::memcpy(resp_visited, new_visited_raws.data(),
              new_visited_raws.size() * sizeof(u64));

  return enqueue_handoff_response(source_shard, std::move(response_msg));
}
