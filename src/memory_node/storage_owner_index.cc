#include "memory_node/memory_node.hh"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#include "common/index_path.hh"
#include "vamana/idmap.hh"
#include "vamana/storage_layout_resolver.hh"

namespace {

using Configuration = configuration::IndexConfiguration;
using BeamEntry = memory_node_detail::BeamEntry;
using HandoffResult = memory_node_detail::HandoffResult;
using HandoffResultStatus = memory_node_detail::HandoffResultStatus;
using NodeSnapshot = memory_node_detail::NodeSnapshot;
using StorageOwnerThread = memory_node_detail::StorageOwnerThread;
using BeamEntrySerialized = service::storage_owner::BeamEntrySerialized;
using SearchHandoffRequestHeader = service::storage_owner::SearchHandoffRequestHeader;
using SearchHandoffResponseHeader = service::storage_owner::SearchHandoffResponseHeader;
using PeerRpcType = service::storage_owner::PeerRpcType;
using InsertStatus = service::storage_owner::InsertStatus;
using InsertBreakdownCounters = service::storage_owner::InsertBreakdownCounters;
static constexpr u32 kPeerRpcMagic = service::storage_owner::kPeerRpcMagic;
using service::storage_owner::handoff_query_vector;
using service::storage_owner::handoff_request_beam;
using service::storage_owner::handoff_request_visited;
using service::storage_owner::handoff_response_beam;
using service::storage_owner::handoff_response_visited;

constexpr size_t kSnapshotPrefixBytes =
  VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE;

size_t snapshot_buffer_bytes() {
  return memory_node_detail::storage_owner_snapshot_bytes();
}

size_t aligned_snapshot_bytes() {
  return memory_node_detail::storage_owner_snapshot_stride();
}

u32 storage_owner_construction_width(const Configuration& config) {
  const u32 configured = config.storage_owner_construction_beam_width == 0
                           ? config.beam_width_construction
                           : config.storage_owner_construction_beam_width;
  return std::max<u32>(1, std::min(config.beam_width_construction, configured));
}

u32 storage_owner_snapshot_batch_size(const Configuration& config,
                                      const StorageOwnerThread* thread = nullptr) {
  const u32 configured = std::max<u32>(1, config.storage_owner_search_snapshot_batch);
  if (thread == nullptr || !thread->has_peer_scratch()) {
    return configured;
  }
  const size_t stride = aligned_snapshot_bytes();
  const size_t capacity = stride == 0 ? 0 : thread->scratch_stride / stride;
  lib_assert(capacity > 0,
             "storage-owner coroutine scratch cannot hold one snapshot: snapshot_stride=" +
             std::to_string(stride) + " scratch_stride=" +
             std::to_string(thread->scratch_stride));
  return static_cast<u32>(std::min<size_t>(configured, capacity));
}

u32 storage_owner_prune_candidate_limit(const Configuration& config) {
  if (config.storage_owner_prune_max_candidates == 0) {
    return std::numeric_limits<u32>::max();
  }
  return std::max(config.R, config.storage_owner_prune_max_candidates);
}

void parse_remote_snapshot(RemotePtr rptr, const byte_t* ptr, NodeSnapshot& snapshot) {
  snapshot = NodeSnapshot{};
  snapshot.rptr = rptr;
  snapshot.header = *reinterpret_cast<const u64*>(ptr);
  snapshot.id = *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_id());
  snapshot.generation = VamanaNode::compact_storage()
    ? *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_generation()) : 0;
  snapshot.deleted = (snapshot.header & VamanaNode::HEADER_DELETED) != 0;
  snapshot.vector_data.resize(VamanaNode::vector_bytes());
  const size_t vector_offset = VamanaNode::compact_storage()
    ? VamanaNode::offset_vector() : kSnapshotPrefixBytes;
  std::memcpy(snapshot.vector_data.data(), ptr + vector_offset, VamanaNode::vector_bytes());
}

struct HandoffTargetShard {
  u32 shard{};
  distance_t best_distance{std::numeric_limits<distance_t>::max()};
};

vec<HandoffTargetShard> collect_handoff_targets(const vec<BeamEntry>& beam,
                                                const vec<byte_t>& shard_searched,
                                                u32 local_shard_id,
                                                u32 shard_count) {
  vec<HandoffTargetShard> targets;
  targets.reserve(shard_count);
  for (const BeamEntry& entry : beam) {
    if (entry.expanded) {
      continue;
    }
    const u32 shard = entry.rptr.memory_node();
    if (shard >= shard_count || shard == local_shard_id || shard_searched[shard]) {
      continue;
    }
    auto it = std::find_if(targets.begin(), targets.end(), [shard](const HandoffTargetShard& target) {
      return target.shard == shard;
    });
    if (it == targets.end()) {
      targets.push_back(HandoffTargetShard{shard, entry.distance});
    } else if (entry.distance < it->best_distance) {
      it->best_distance = entry.distance;
    }
  }
  std::sort(targets.begin(), targets.end(), [](const HandoffTargetShard& lhs, const HandoffTargetShard& rhs) {
    return lhs.best_distance < rhs.best_distance;
  });
  return targets;
}

void mark_shard_frontier_expanded(vec<BeamEntry>& beam, u32 shard) {
  for (BeamEntry& entry : beam) {
    if (!entry.expanded && entry.rptr.memory_node() == shard) {
      entry.expanded = true;
    }
  }
}

vec<BeamEntrySerialized> serialize_unexpanded_beam(const vec<BeamEntry>& beam) {
  vec<BeamEntrySerialized> serialized;
  serialized.reserve(beam.size());
  for (const BeamEntry& entry : beam) {
    if (!entry.expanded) {
      serialized.push_back({entry.rptr.raw_address, entry.distance});
    }
  }
  return serialized;
}

vec<u64> serialize_visited_for_shard(const hashset_t<RemotePtr>& visited,
                                     u32 target_shard,
                                     size_t max_visited) {
  vec<u64> serialized;
  serialized.reserve(std::min(visited.size(), max_visited));
  for (const RemotePtr& entry : visited) {
    if (entry.memory_node() == target_shard) {
      serialized.push_back(entry.raw_address);
      if (serialized.size() == max_visited) {
        break;
      }
    }
  }
  return serialized;
}

vec<byte_t> build_search_handoff_request(const span<const element_t> query,
                                         const vec<BeamEntry>& beam,
                                         const hashset_t<RemotePtr>& visited,
                                         u32 target_shard,
                                         u32 source_shard,
                                         u64 request_id,
                                         const Configuration& config) {
  const vec<BeamEntrySerialized> beam_serialized = serialize_unexpanded_beam(beam);
  const size_t max_visited = static_cast<size_t>(storage_owner_construction_width(config)) * config.R;
  const vec<u64> visited_serialized = serialize_visited_for_shard(visited, target_shard, max_visited);
  const u32 vector_bytes = static_cast<u32>(VamanaNode::DIM * sizeof(element_t));
  const size_t payload_bytes = static_cast<size_t>(vector_bytes) +
                               beam_serialized.size() * sizeof(BeamEntrySerialized) +
                               visited_serialized.size() * sizeof(u64);

  vec<byte_t> msg_buffer(sizeof(SearchHandoffRequestHeader) + payload_bytes);
  auto* req = reinterpret_cast<SearchHandoffRequestHeader*>(msg_buffer.data());
  req->rpc.magic = kPeerRpcMagic;
  req->rpc.type = static_cast<u32>(PeerRpcType::search_handoff_request);
  req->rpc.source_shard = source_shard;
  req->rpc.item_count = static_cast<u32>(beam_serialized.size());
  req->rpc.request_id = request_id;
  req->rpc.status = static_cast<u32>(InsertStatus::failed);
  req->rpc.reserved = 0;
  req->beam_width = storage_owner_construction_width(config);
  req->snapshot_batch = storage_owner_snapshot_batch_size(config);
  req->originator_shard = source_shard;
  req->visited_count = static_cast<u32>(visited_serialized.size());
  req->vector_bytes = vector_bytes;
  req->reserved = 0;

  std::memcpy(handoff_query_vector(req), query.data(), vector_bytes);
  auto* beam_out = handoff_request_beam(req, vector_bytes);
  std::memcpy(beam_out, beam_serialized.data(), beam_serialized.size() * sizeof(BeamEntrySerialized));
  auto* visited_out = handoff_request_visited(req, vector_bytes, static_cast<u32>(beam_serialized.size()));
  std::memcpy(visited_out, visited_serialized.data(), visited_serialized.size() * sizeof(u64));
  return msg_buffer;
}

void merge_or_update_beam_entry(vec<BeamEntry>& beam, RemotePtr rptr, distance_t dist, u32 max_beam_width) {
  for (BeamEntry& entry : beam) {
    if (entry.rptr == rptr) {
      if (dist < entry.distance) {
        entry.distance = dist;
      }
      return;
    }
  }

  auto it = std::lower_bound(
    beam.begin(), beam.end(), dist, [](const BeamEntry& entry, distance_t value) { return entry.distance < value; });
  beam.insert(it, BeamEntry{rptr, dist, false});
  if (beam.size() > max_beam_width) {
    beam.resize(max_beam_width);
  }
}

bool merge_search_handoff_response(vec<BeamEntry>& beam,
                                   hashset_t<RemotePtr>& visited,
                                   const HandoffResult& result,
                                   u32 construction_width) {
  if (!result.ok() || result.response.empty()) {
    return false;
  }
  const auto* resp = reinterpret_cast<const SearchHandoffResponseHeader*>(result.response.data());
  if (resp->rpc.magic != kPeerRpcMagic ||
      resp->rpc.type != static_cast<u32>(PeerRpcType::search_handoff_response) ||
      resp->rpc.status != static_cast<u32>(InsertStatus::ok)) {
    return false;
  }

  const auto* resp_beam = handoff_response_beam(resp);
  for (u32 i = 0; i < resp->updated_beam_count; ++i) {
    merge_or_update_beam_entry(beam, RemotePtr{resp_beam[i].rptr_raw}, resp_beam[i].distance, construction_width);
  }

  const byte_t* resp_visited = handoff_response_visited(resp, resp->updated_beam_count);
  for (u32 i = 0; i < resp->new_visited_count; ++i) {
    u64 raw{};
    std::memcpy(&raw, resp_visited + static_cast<size_t>(i) * sizeof(raw), sizeof(raw));
    visited.insert(RemotePtr{raw});
  }
  return true;
}

void record_search_handoff_stats(InsertBreakdownCounters* breakdown,
                                 const HandoffResult& result,
                                 size_t request_bytes) {
  if (breakdown == nullptr) {
    return;
  }
  ++breakdown->storage_owner_handoff_requests;
  breakdown->storage_owner_handoff_request_bytes += request_bytes;
  breakdown->storage_owner_handoff_response_bytes += result.response.size();

  switch (result.status) {
    case HandoffResultStatus::ok:
      ++breakdown->storage_owner_handoff_successes;
      break;
    case HandoffResultStatus::queue_full:
      ++breakdown->storage_owner_handoff_queue_full;
      break;
    case HandoffResultStatus::timeout:
      ++breakdown->storage_owner_handoff_timeouts;
      break;
    case HandoffResultStatus::overloaded:
      ++breakdown->storage_owner_handoff_overloaded;
      break;
    case HandoffResultStatus::pending:
    case HandoffResultStatus::shutdown:
    case HandoffResultStatus::failed:
      ++breakdown->storage_owner_handoff_failed;
      break;
  }

  if (!result.ok() || result.response.size() < sizeof(SearchHandoffResponseHeader)) {
    return;
  }
  const auto* resp = reinterpret_cast<const SearchHandoffResponseHeader*>(result.response.data());
  if (resp->rpc.magic != kPeerRpcMagic ||
      resp->rpc.type != static_cast<u32>(PeerRpcType::search_handoff_response) ||
      resp->rpc.status != static_cast<u32>(InsertStatus::ok)) {
    return;
  }
  breakdown->storage_owner_handoff_remote_handler_ns += resp->handler_cpu_ns;
  breakdown->storage_owner_handoff_remote_expanded_nodes += resp->local_expanded_count;
  breakdown->storage_owner_handoff_remote_snapshot_reads += resp->local_snapshot_reads;
  breakdown->storage_owner_handoff_remote_neighbor_reads += resp->local_neighbor_reads;
  breakdown->storage_owner_handoff_response_beam_entries += resp->updated_beam_count;
  breakdown->storage_owner_handoff_response_visited_entries += resp->new_visited_count;
  breakdown->storage_owner_handoff_response_visited_truncated += resp->visited_truncated_count;
}

}  // namespace

RemotePtr MemoryNode::allocate_local_node() {
  size_t node_size = VamanaNode::allocation_size();
  while (node_size % 8 != 0) {
    node_size += 4;
  }

  auto* free_ptr = reinterpret_cast<u64*>(index_buffer_.get_full_buffer());
  std::atomic_ref<u64> alloc_ref(*free_ptr);
  const u64 offset = alloc_ref.fetch_add(node_size, std::memory_order_acq_rel);
  lib_assert(offset + node_size <= mn_memory_bytes_, "storage node out of memory");
  return RemotePtr{storage_id_, offset};
}

bool MemoryNode::load_owner_idmap(const filepath_t& index_prefix) {
  idmap_.clear();
  mutations_inflight_.clear();
  if (index_prefix.empty()) {
    return true;
  }
  const filepath_t path = index_path::owner_idmap_file(index_prefix, storage_id_ + 1, num_storage_nodes_);
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    std::cerr << "[storage-owner] missing idmap sidecar: " << path << std::endl;
    return false;
  }
  vamana::idmap::Header header;
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!input.good() || header.magic != vamana::idmap::kMagic ||
      header.version != vamana::idmap::kVersion ||
      header.owner_shard != storage_id_ ||
      header.shard_count != num_storage_nodes_) {
    std::cerr << "[storage-owner] invalid idmap sidecar: " << path << std::endl;
    return false;
  }
  idmap_.reserve(static_cast<size_t>(header.entry_count));
  for (u64 i = 0; i < header.entry_count; ++i) {
    vamana::idmap::Entry entry;
    input.read(reinterpret_cast<char*>(&entry), sizeof(entry));
    if (!input.good()) return false;
    idmap_[entry.id] = FreshnessEntry{
      RemotePtr{entry.rptr_raw},
      entry.generation,
      (entry.flags & vamana::idmap::kDeleted) != 0};
  }
  return true;
}

void MemoryNode::publish_mutation(node_t id, RemotePtr ptr, u32 generation, bool deleted) {
  std::lock_guard<std::mutex> lock(idmap_mutex_);
  idmap_[id] = FreshnessEntry{ptr, generation, deleted};
  mutations_inflight_.erase(id);
}

service::storage_owner::MutationStatus MemoryNode::prepare_mutation(
    node_t id,
    service::storage_owner::MutationKind kind,
    FreshnessEntry* old_entry,
    u32* new_generation) {
  std::lock_guard<std::mutex> lock(idmap_mutex_);
  if (mutations_inflight_.contains(id)) {
    return service::storage_owner::MutationStatus::failed;
  }
  auto it = idmap_.find(id);
  const bool exists = it != idmap_.end();
  const bool live = exists && !it->second.deleted;
  if (old_entry != nullptr) {
    *old_entry = exists ? it->second : FreshnessEntry{};
  }
  const u32 previous_generation = exists ? it->second.generation : 0;
  if (new_generation != nullptr) {
    *new_generation = previous_generation + 1;
  }
  if (kind == service::storage_owner::MutationKind::insert && live) {
    return service::storage_owner::MutationStatus::already_exists;
  }
  if (kind == service::storage_owner::MutationKind::erase) {
    if (!exists) return service::storage_owner::MutationStatus::not_found;
    if (!live) return service::storage_owner::MutationStatus::already_deleted;
  }
  mutations_inflight_.insert(id);
  return service::storage_owner::MutationStatus::ok;
}

bool MemoryNode::mark_node_deleted(RemotePtr rptr, u32 generation) {
  if (rptr.is_null()) return true;
  const auto header_addr = vamana::StorageLayoutResolver::header(rptr);
  const bool remote = !local_shard(rptr.memory_node());
  if (local_shard(rptr.memory_node())) {
    auto* header_ptr = reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + header_addr.offset);
    std::atomic_ref<u64> ref(*header_ptr);
    ref.fetch_or(static_cast<u64>(VamanaNode::HEADER_DELETED), std::memory_order_acq_rel);
  } else {
    lock_node(rptr);
    u64 header = 0;
    remote_read_bytes(rptr.memory_node(), header_addr.offset, &header, sizeof(header), 0);
    header |= static_cast<u64>(VamanaNode::HEADER_DELETED);
    remote_write_bytes(rptr.memory_node(), header_addr.offset, &header, sizeof(header), 0);
  }
  if (VamanaNode::compact_storage()) {
    vec<byte_t> entry(VamanaNode::hot_graph_entry_size(), 0);
    VamanaNode::encode_hot_graph_entry(entry.data(), 0, 0, nullptr, 0,
      VamanaNode::HOT_GRAPH_SHARD_BITS, generation, 2, true);
    const u64 hot_offset = VamanaNode::hot_graph_entry_offset(rptr);
    if (local_shard(rptr.memory_node())) {
      std::memcpy(index_buffer_.get_full_buffer() + hot_offset, entry.data(), entry.size());
    } else {
      remote_write_bytes(rptr.memory_node(), hot_offset, entry.data(), entry.size(), 0);
    }
  }
  if (remote) {
    unlock_node(rptr);
  }
  return true;
}

RemotePtr MemoryNode::read_global_medoid() {
  if (storage_id_ == 0) {
    return RemotePtr{*reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + 8)};
  }

  u64 raw = 0;
  remote_read_bytes(0, 8, &raw, sizeof(raw), 0);
  return RemotePtr{raw};
}

auto MemoryNode::async_read_global_medoid(StorageOwnerThread& thread) {
  struct Awaitable {
    bool ready{};
    byte_t* buffer{};
    MemoryNode* node{};

    bool await_ready() const { return ready; }
    static void await_suspend(std::coroutine_handle<>) {}
    RemotePtr await_resume() const {
      if (node->storage_id_ == 0) {
        return RemotePtr{*reinterpret_cast<u64*>(node->index_buffer_.get_full_buffer() + 8)};
      }
      return RemotePtr{*reinterpret_cast<const u64*>(buffer)};
    }
  };

  if (storage_id_ == 0) {
    return Awaitable{true, nullptr, this};
  }
  byte_t* buffer = thread.coroutine_scratch();
  post_peer_read_async(thread, 0, 8, buffer, sizeof(u64));
  return Awaitable{false, buffer, this};
}

void MemoryNode::write_global_medoid(const RemotePtr& medoid) {
  if (storage_id_ == 0) {
    *reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + 8) = medoid.raw_address;
    return;
  }
  remote_write_bytes(0, 8, &medoid.raw_address, sizeof(medoid.raw_address), 0);
}

bool MemoryNode::try_set_global_medoid(const RemotePtr& expected, const RemotePtr& desired, RemotePtr& observed) {
  if (storage_id_ == 0) {
    auto* slot = reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + 8);
    std::atomic_ref<u64> ref(*slot);
    u64 current = expected.raw_address;
    const bool ok =
      ref.compare_exchange_strong(current, desired.raw_address, std::memory_order_acq_rel, std::memory_order_acquire);
    observed = RemotePtr{current};
    return ok;
  }

  const u64 original = remote_compare_and_swap(0, 8, expected.raw_address, desired.raw_address, 0);
  observed = RemotePtr{original};
  return original == expected.raw_address;
}

bool MemoryNode::read_node_snapshot(RemotePtr rptr, NodeSnapshot& snapshot) {
  lib_assert(rptr.memory_node() < num_storage_nodes_,
             "invalid remote shard id in read_node_snapshot: " + std::to_string(rptr.memory_node()));
  const auto vector_addr = vamana::StorageLayoutResolver::vector(rptr);
  lib_assert(vector_addr.offset + vector_addr.size <= mn_memory_bytes_,
             "node snapshot read exceeds shard bounds: shard=" + std::to_string(rptr.memory_node()) +
               " offset=" + std::to_string(rptr.byte_offset()) +
               " size=" + std::to_string(vector_addr.size) +
               " capacity=" + std::to_string(mn_memory_bytes_));
  snapshot = NodeSnapshot{};
  snapshot.rptr = rptr;
  snapshot.vector_data.resize(VamanaNode::vector_bytes());

  const size_t read_size = VamanaNode::vector_bytes();
  if (local_shard(rptr.memory_node())) {
    const byte_t* base = index_buffer_.get_full_buffer();
    const byte_t* ptr = base + rptr.byte_offset();
    snapshot.header = *reinterpret_cast<const u64*>(ptr);
    snapshot.id = *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_id());
    snapshot.generation = VamanaNode::compact_storage()
      ? *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_generation()) : 0;
    snapshot.edge_count = VamanaNode::compact_storage()
      ? 0 : *reinterpret_cast<const u8*>(ptr + VamanaNode::offset_edge_count());
    snapshot.deleted = (snapshot.header & VamanaNode::HEADER_DELETED) != 0;
    std::memcpy(snapshot.vector_data.data(), base + vector_addr.offset, VamanaNode::vector_bytes());
    return true;
  }

  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  byte_t* read_buffer = owner_thread != nullptr && owner_thread->has_peer_scratch()
                          ? owner_thread->scratch_buffer.get_full_buffer()
                          : peer_scratch_buffer_.get_full_buffer();
  byte_t prefix[VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE]{};
  remote_read_bytes(rptr.memory_node(), rptr.byte_offset(), prefix, sizeof(prefix), 0);
  remote_read_bytes(rptr.memory_node(),
                    vector_addr.offset,
                    read_buffer,
                    read_size,
                    0);
  snapshot.vector_data.resize(VamanaNode::vector_bytes());
  std::memcpy(snapshot.vector_data.data(), read_buffer, VamanaNode::vector_bytes());
  snapshot.header = *reinterpret_cast<const u64*>(prefix);
  snapshot.id = *reinterpret_cast<const u32*>(prefix + VamanaNode::offset_id());
  snapshot.generation = VamanaNode::compact_storage()
    ? *reinterpret_cast<const u32*>(prefix + VamanaNode::offset_generation()) : 0;
  snapshot.edge_count = VamanaNode::compact_storage()
    ? 0 : *reinterpret_cast<const u8*>(prefix + VamanaNode::offset_edge_count());
  snapshot.deleted = (snapshot.header & VamanaNode::HEADER_DELETED) != 0;
  return true;
}

vec<RemotePtr> MemoryNode::read_neighbor_list_aos(RemotePtr rptr) {
  lib_assert(rptr.memory_node() < num_storage_nodes_,
             "invalid remote shard id in read_neighbor_list: " + std::to_string(rptr.memory_node()));
  const auto neighbor_addr = vamana::StorageLayoutResolver::neighbor_slots(rptr);
  lib_assert(neighbor_addr.offset + neighbor_addr.size <= mn_memory_bytes_,
             "neighbor-list read exceeds shard bounds: shard=" + std::to_string(rptr.memory_node()) +
               " offset=" + std::to_string(rptr.byte_offset()) +
               " size=" + std::to_string(neighbor_addr.size) +
               " capacity=" + std::to_string(mn_memory_bytes_));
  vec<RemotePtr> neighbors;
  if (local_shard(rptr.memory_node())) {
    const byte_t* ptr = local_node_ptr(rptr);
    const u8 edge_count = *reinterpret_cast<const u8*>(ptr + VamanaNode::offset_edge_count());
    const auto* slots = reinterpret_cast<const RemotePtr*>(ptr + VamanaNode::offset_neighbors());
    neighbors.reserve(edge_count);
    for (u32 i = 0; i < edge_count; ++i) {
      if (!slots[i].is_null()) {
        neighbors.push_back(slots[i]);
      }
    }
    return neighbors;
  }

  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  byte_t* read_buffer = owner_thread != nullptr && owner_thread->has_peer_scratch()
                          ? owner_thread->scratch_buffer.get_full_buffer()
                          : peer_scratch_buffer_.get_full_buffer();
  remote_read_bytes(rptr.memory_node(),
                    vamana::StorageLayoutResolver::neighbor_read(rptr).address.offset,
                    read_buffer,
                    VamanaNode::neighbor_read_size(),
                    0);
  const u8 edge_count = *reinterpret_cast<const u8*>(read_buffer + VamanaNode::neighbor_count_offset_in_read());
  const auto* slots = reinterpret_cast<const RemotePtr*>(read_buffer + VamanaNode::neighbor_payload_offset_in_read());
  neighbors.reserve(edge_count);
  for (u32 i = 0; i < edge_count && i < VamanaNode::R; ++i) {
    if (!slots[i].is_null()) {
      neighbors.push_back(slots[i]);
    }
  }
  return neighbors;
}

vec<RemotePtr> MemoryNode::read_neighbor_list(RemotePtr rptr) {
  if (!VamanaNode::compact_storage()) {
    return read_neighbor_list_aos(rptr);
  }

  vec<byte_t> local_entry;
  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  byte_t* read_buffer = nullptr;
  if (local_shard(rptr.memory_node())) {
    local_entry.resize(VamanaNode::hot_graph_entry_size());
    read_buffer = local_entry.data();
  } else {
    read_buffer = owner_thread != nullptr && owner_thread->has_peer_scratch()
                    ? owner_thread->scratch_buffer.get_full_buffer()
                    : peer_scratch_buffer_.get_full_buffer();
  }
  vec<byte_t> decoded(VamanaNode::neighbor_read_size());
  bool decoded_ok = false;
  constexpr u32 kMaxReadAttempts = 3;
  for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
    if (local_shard(rptr.memory_node())) {
      std::memcpy(read_buffer,
                  index_buffer_.get_full_buffer() + VamanaNode::hot_graph_entry_offset(rptr),
                  VamanaNode::hot_graph_entry_size());
    } else {
      remote_read_bytes(rptr.memory_node(),
                        VamanaNode::hot_graph_entry_offset(rptr),
                        read_buffer,
                        VamanaNode::hot_graph_entry_size(),
                        0);
    }
    decoded_ok = VamanaNode::decode_hot_graph_entry(read_buffer, decoded.data());
    if (decoded_ok) {
      break;
    }
    std::this_thread::yield();
  }
  if (!decoded_ok) {
    return {};
  }
  const byte_t* parse_buffer = decoded.data();
  const u8 edge_count = *reinterpret_cast<const u8*>(parse_buffer + VamanaNode::neighbor_count_offset_in_read());
  const auto* slots = reinterpret_cast<const RemotePtr*>(parse_buffer + VamanaNode::neighbor_payload_offset_in_read());
  vec<RemotePtr> neighbors;
  neighbors.reserve(edge_count);
  for (u32 i = 0; i < edge_count && i < VamanaNode::R; ++i) {
    if (!slots[i].is_null()) {
      neighbors.push_back(slots[i]);
    }
  }
  return neighbors;
}

auto MemoryNode::async_read_node_snapshot(RemotePtr rptr, StorageOwnerThread& thread) {
  struct Awaitable {
    bool ready{};
    RemotePtr rptr;
    byte_t* buffer{};
    NodeSnapshot snapshot;
    MemoryNode* node{};
    StorageOwnerThread* thread{};

    bool await_ready() const { return ready; }
    static void await_suspend(std::coroutine_handle<>) {}
    NodeSnapshot await_resume() {
      if (ready) {
        return std::move(snapshot);
      }
      parse_remote_snapshot(rptr, buffer, snapshot);
      return std::move(snapshot);
    }
  };

  if (local_shard(rptr.memory_node())) {
    NodeSnapshot snapshot;
    read_node_snapshot(rptr, snapshot);
    return Awaitable{true, rptr, nullptr, std::move(snapshot), this, &thread};
  }

  byte_t* buffer = thread.coroutine_scratch();
  if (VamanaNode::compact_storage()) {
    post_peer_read_async(thread, rptr.memory_node(), rptr.byte_offset(), buffer,
                         VamanaNode::size_until_vector_end());
  } else {
    post_peer_read_async(thread, rptr.memory_node(), rptr.byte_offset(), buffer,
                         kSnapshotPrefixBytes);
    post_peer_read_async(thread, rptr.memory_node(),
                         vamana::StorageLayoutResolver::vector(rptr).offset,
                         buffer + kSnapshotPrefixBytes, VamanaNode::vector_bytes());
  }
  return Awaitable{false, rptr, buffer, {}, this, &thread};
}

auto MemoryNode::async_read_node_snapshots(const vec<RemotePtr>& rptrs,
                                           const Configuration& config,
                                           StorageOwnerThread& thread) {
  struct PendingRead {
    RemotePtr rptr;
    byte_t* buffer{};
  };

  struct Awaitable {
    bool ready{true};
    vec<NodeSnapshot> snapshots;
    vec<PendingRead> pending;
    StorageOwnerThread* thread{};

    bool await_ready() const { return ready; }
    static void await_suspend(std::coroutine_handle<>) {}
    vec<NodeSnapshot> await_resume() {
      for (const PendingRead& read : pending) {
        NodeSnapshot snapshot;
        parse_remote_snapshot(read.rptr, read.buffer, snapshot);
        snapshots.push_back(std::move(snapshot));
      }
      return std::move(snapshots);
    }
  };

  Awaitable awaitable;
  awaitable.snapshots.reserve(rptrs.size());
  awaitable.pending.reserve(rptrs.size());
  awaitable.thread = &thread;

  const size_t snapshot_size = snapshot_buffer_bytes();
  const size_t snapshot_stride = aligned_snapshot_bytes();
  const u32 max_batch = storage_owner_snapshot_batch_size(config, &thread);
  lib_assert(rptrs.size() <= max_batch, "storage-owner snapshot batch exceeds configured limit");

  u32 remote_slot = 0;
  for (const RemotePtr& rptr : rptrs) {
    if (rptr.is_null()) {
      continue;
    }

    NodeSnapshot snapshot;
    if (local_shard(rptr.memory_node())) {
      read_node_snapshot(rptr, snapshot);
      awaitable.snapshots.push_back(std::move(snapshot));
      continue;
    }

    const size_t scratch_offset = static_cast<size_t>(remote_slot) * snapshot_stride;
    lib_assert(scratch_offset + snapshot_size <= thread.scratch_stride,
               "storage-owner coroutine scratch stride is too small for snapshot batch: "
               "offset=" + std::to_string(scratch_offset) +
               " snapshot=" + std::to_string(snapshot_size) +
               " stride=" + std::to_string(thread.scratch_stride) +
               " remote_slot=" + std::to_string(remote_slot) +
               " batch=" + std::to_string(rptrs.size()));
    byte_t* buffer = thread.coroutine_scratch(scratch_offset);
    if (VamanaNode::compact_storage()) {
      post_peer_read_async(thread, rptr.memory_node(), rptr.byte_offset(), buffer,
                           VamanaNode::size_until_vector_end());
    } else {
      post_peer_read_async(thread, rptr.memory_node(), rptr.byte_offset(), buffer,
                           kSnapshotPrefixBytes);
      post_peer_read_async(thread, rptr.memory_node(),
                           vamana::StorageLayoutResolver::vector(rptr).offset,
                           buffer + kSnapshotPrefixBytes, VamanaNode::vector_bytes());
    }
    awaitable.pending.push_back(PendingRead{rptr, buffer});
    awaitable.ready = false;
    ++remote_slot;
  }

  return awaitable;
}

vec<MemoryNode::NodeSnapshot> MemoryNode::read_node_snapshots_batched(const vec<RemotePtr>& rptrs,
                                                                      const Configuration& config) {
  vec<NodeSnapshot> snapshots;
  snapshots.reserve(rptrs.size());
  if (rptrs.empty()) {
    return snapshots;
  }

  StorageOwnerThread* thread = current_storage_owner_thread_;
  if (thread == nullptr || !thread->has_peer_scratch()) {
    for (const RemotePtr& rptr : rptrs) {
      NodeSnapshot snapshot;
      if (!rptr.is_null() && read_node_snapshot(rptr, snapshot)) {
        snapshots.push_back(std::move(snapshot));
      }
    }
    return snapshots;
  }

  struct PendingRead {
    RemotePtr rptr;
    byte_t* buffer{};
  };

  const size_t snapshot_size = snapshot_buffer_bytes();
  const size_t snapshot_stride = aligned_snapshot_bytes();
  const size_t max_batch = storage_owner_snapshot_batch_size(config, thread);

  for (size_t begin = 0; begin < rptrs.size(); begin += max_batch) {
    const size_t end = std::min(rptrs.size(), begin + max_batch);
    vec<PendingRead> pending;
    pending.reserve(end - begin);
    u32 remote_slot = 0;

    for (size_t idx = begin; idx < end; ++idx) {
      const RemotePtr& rptr = rptrs[idx];
      if (rptr.is_null()) {
        continue;
      }

      NodeSnapshot snapshot;
      if (local_shard(rptr.memory_node())) {
        read_node_snapshot(rptr, snapshot);
        snapshots.push_back(std::move(snapshot));
        continue;
      }

      const size_t scratch_offset = static_cast<size_t>(remote_slot) * snapshot_stride;
      lib_assert(scratch_offset + snapshot_size <= thread->scratch_stride,
                 "storage-owner coroutine scratch stride is too small for snapshot batch: "
                 "offset=" + std::to_string(scratch_offset) +
                 " snapshot=" + std::to_string(snapshot_size) +
                 " stride=" + std::to_string(thread->scratch_stride) +
                 " remote_slot=" + std::to_string(remote_slot) +
                 " chunk=" + std::to_string(end - begin));
      byte_t* buffer = thread->coroutine_scratch(scratch_offset);
      if (VamanaNode::compact_storage()) {
        post_peer_read_async(*thread, rptr.memory_node(), rptr.byte_offset(), buffer,
                             VamanaNode::size_until_vector_end());
      } else {
        post_peer_read_async(*thread, rptr.memory_node(), rptr.byte_offset(), buffer,
                             kSnapshotPrefixBytes);
        post_peer_read_async(*thread, rptr.memory_node(),
                             vamana::StorageLayoutResolver::vector(rptr).offset,
                             buffer + kSnapshotPrefixBytes, VamanaNode::vector_bytes());
      }
      pending.push_back(PendingRead{rptr, buffer});
      ++remote_slot;
    }

    while (!thread->is_ready(thread->running_coroutine)) {
      poll_peer_send_cq();
      std::this_thread::yield();
    }

    for (const PendingRead& read : pending) {
      NodeSnapshot snapshot;
      parse_remote_snapshot(read.rptr, read.buffer, snapshot);
      snapshots.push_back(std::move(snapshot));
    }
  }

  return snapshots;
}

auto MemoryNode::async_read_neighbor_list(RemotePtr rptr, StorageOwnerThread& thread) {
  struct Awaitable {
    bool ready{};
    RemotePtr rptr;
    byte_t* buffer{};
    vec<RemotePtr> neighbors;
    MemoryNode* node{};
    StorageOwnerThread* thread{};
    bool hot_graph{};

    bool await_ready() const { return ready; }
    static void await_suspend(std::coroutine_handle<>) {}
    vec<RemotePtr> await_resume() {
      if (ready) {
        return std::move(neighbors);
      }
      vec<byte_t> decoded;
      const byte_t* parse_buffer = buffer;
      if (hot_graph) {
        decoded.resize(VamanaNode::neighbor_read_size());
        const bool ok = VamanaNode::decode_hot_graph_entry(buffer, decoded.data());
        if (!ok) {
          return node->read_neighbor_list(rptr);
        }
        parse_buffer = decoded.data();
      }
      const u8 edge_count = *reinterpret_cast<const u8*>(
        parse_buffer + VamanaNode::neighbor_count_offset_in_read());
      const auto* slots = reinterpret_cast<const RemotePtr*>(
        parse_buffer + VamanaNode::neighbor_payload_offset_in_read());
      neighbors.reserve(edge_count);
      for (u32 i = 0; i < edge_count && i < VamanaNode::R; ++i) {
        if (!slots[i].is_null()) {
          neighbors.push_back(slots[i]);
        }
      }
      return std::move(neighbors);
    }
  };

  if (local_shard(rptr.memory_node())) {
    vec<RemotePtr> neighbors = read_neighbor_list(rptr);
    return Awaitable{true, rptr, nullptr, std::move(neighbors), this, &thread, false};
  }

  byte_t* buffer = thread.coroutine_scratch();
  const auto neighbor_read = vamana::StorageLayoutResolver::neighbor_read(rptr);
  const bool use_hot_graph = neighbor_read.compact;
  post_peer_read_async(thread,
                       rptr.memory_node(),
                       neighbor_read.address.offset,
                       buffer,
                       neighbor_read.address.size);
  return Awaitable{false, rptr, buffer, {}, this, &thread, use_hot_graph};
}

void MemoryNode::write_hot_graph_entry(RemotePtr rptr, u32 id, const vec<RemotePtr>& neighbors) {
  if (!VamanaNode::hot_graph_entry_available(rptr)) {
    return;
  }
  const size_t entry_size = VamanaNode::hot_graph_entry_size();
  vec<byte_t> entry(entry_size, 0);
  const u8 edge_count = static_cast<u8>(std::min<size_t>(neighbors.size(), VamanaNode::R));
  VamanaNode::encode_hot_graph_entry(entry.data(), id, edge_count,
                                     neighbors.data(), edge_count);
  const u64 hot_offset = VamanaNode::hot_graph_entry_offset(rptr);
  if (local_shard(rptr.memory_node())) {
    lib_assert(hot_offset + entry_size <= mn_memory_bytes_,
               "hot graph write exceeds shard bounds");
    std::memcpy(index_buffer_.get_full_buffer() + hot_offset, entry.data(), entry_size);
    return;
  }
  remote_write_bytes(rptr.memory_node(), hot_offset, entry.data(), entry_size, 0);
}

void MemoryNode::write_neighbor_list(RemotePtr rptr, const vec<RemotePtr>& neighbors) {
  lib_assert(rptr.memory_node() < num_storage_nodes_,
             "invalid remote shard id in write_neighbor_list: " + std::to_string(rptr.memory_node()));
  const auto neighbor_addr = vamana::StorageLayoutResolver::neighbor_slots(rptr);
  lib_assert(neighbor_addr.offset + neighbor_addr.size <= mn_memory_bytes_,
             "neighbor-list write exceeds shard bounds: shard=" + std::to_string(rptr.memory_node()) +
               " offset=" + std::to_string(rptr.byte_offset()) +
               " size=" + std::to_string(neighbor_addr.size) +
               " capacity=" + std::to_string(mn_memory_bytes_));
  const u8 edge_count = static_cast<u8>(std::min<size_t>(neighbors.size(), VamanaNode::R));
  if (VamanaNode::compact_storage()) {
    write_hot_graph_entry(rptr, 0, neighbors);
    return;
  }
  if (local_shard(rptr.memory_node())) {
    byte_t* ptr = local_node_ptr(rptr);
    const u32 id = *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_id());
    *reinterpret_cast<u8*>(ptr + VamanaNode::offset_edge_count()) = edge_count;
    std::memset(ptr + VamanaNode::offset_edge_count() + sizeof(u8), 0, VamanaNode::PADDING_SIZE);
    auto* slots = reinterpret_cast<RemotePtr*>(ptr + VamanaNode::offset_neighbors());
    for (u32 i = 0; i < edge_count; ++i) {
      slots[i] = neighbors[i];
    }
    for (u32 i = edge_count; i < VamanaNode::R; ++i) {
      slots[i].reset();
    }
    write_hot_graph_entry(rptr, id, neighbors);
    return;
  }

  byte_t meta[sizeof(u8) + VamanaNode::PADDING_SIZE]{};
  meta[0] = edge_count;
  remote_write_bytes(rptr.memory_node(), rptr.byte_offset() + VamanaNode::offset_edge_count(), meta, sizeof(meta), 0);

  vec<RemotePtr> slots(VamanaNode::R);
  for (u32 i = 0; i < edge_count; ++i) {
    slots[i] = neighbors[i];
  }
  remote_write_bytes(rptr.memory_node(),
                     rptr.byte_offset() + VamanaNode::offset_neighbors(),
                     slots.data(),
                     VamanaNode::NEIGHBORS_SIZE,
                     align_up(sizeof(meta)));
  write_hot_graph_entry(rptr, 0, neighbors);
}

void MemoryNode::write_new_node(RemotePtr rptr,
                    node_t id,
                    const span<const element_t> components,
                    const vec<RemotePtr>& neighbors,
                    u32 generation) {
  byte_t* ptr = local_node_ptr(rptr);
  std::memset(ptr, 0, VamanaNode::total_size());
  *reinterpret_cast<u64*>(ptr) = 0;
  *reinterpret_cast<u32*>(ptr + VamanaNode::offset_id()) = id;
  if (VamanaNode::compact_storage()) {
    *reinterpret_cast<u32*>(ptr + VamanaNode::offset_generation()) = generation;
  } else {
    *reinterpret_cast<u8*>(ptr + VamanaNode::offset_edge_count()) =
      static_cast<u8>(std::min<size_t>(neighbors.size(), VamanaNode::R));
  }
  encode_float_vector_to_storage(components.data(), VamanaNode::DIM, VamanaNode::vector_dtype(),
                                 ptr + VamanaNode::offset_vector());
  if (!VamanaNode::compact_storage()) {
    auto* slots = reinterpret_cast<RemotePtr*>(ptr + VamanaNode::offset_neighbors());
    for (u32 i = 0; i < neighbors.size() && i < VamanaNode::R; ++i) {
      slots[i] = neighbors[i];
    }
  }
  write_hot_graph_entry(rptr, id, neighbors);
  if (VamanaNode::HAS_RABITQ_CODE) {
      VamanaNode::RabitqCode code;
      float norm = 0.0f;
      float error = 0.0f;
      VamanaNode::compute_rabitq_entry(
          ptr + VamanaNode::offset_vector(), VamanaNode::vector_dtype(), code, norm, error);
      std::memcpy(ptr + VamanaNode::offset_rabitq_code(), code.data(), code.size());
      *reinterpret_cast<float*>(ptr + VamanaNode::offset_rabitq_norm()) = norm;
      *reinterpret_cast<float*>(ptr + VamanaNode::offset_rabitq_error()) = error;
  }
}

void MemoryNode::lock_node(RemotePtr rptr) {
  if (local_shard(rptr.memory_node())) {
    auto* header_ptr = reinterpret_cast<u64*>(
      index_buffer_.get_full_buffer() + vamana::StorageLayoutResolver::header(rptr).offset);
    std::atomic_ref<u64> ref(*header_ptr);
    for (;;) {
      u64 header = ref.load(std::memory_order_acquire);
      if ((header & VamanaNode::HEADER_NODE_LOCK) != 0) {
        std::this_thread::yield();
        continue;
      }
      const u64 desired = header | VamanaNode::HEADER_NODE_LOCK;
      if (ref.compare_exchange_weak(header, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return;
      }
    }
  }

  for (;;) {
    auto [success, header] = try_lock_remote_header(rptr);
    if (success) {
      return;
    }
    if ((header & VamanaNode::HEADER_NODE_LOCK) != 0) {
      std::this_thread::yield();
    }
  }
}

void MemoryNode::unlock_node(RemotePtr rptr) {
  if (local_shard(rptr.memory_node())) {
    auto* header_ptr = reinterpret_cast<u64*>(
      index_buffer_.get_full_buffer() + vamana::StorageLayoutResolver::header(rptr).offset);
    std::atomic_ref<u64> ref(*header_ptr);
    ref.fetch_and(~static_cast<u64>(VamanaNode::HEADER_NODE_LOCK), std::memory_order_acq_rel);
    return;
  }

  const byte_t unlock = 0;
  remote_write_bytes(rptr.memory_node(),
                     vamana::StorageLayoutResolver::header(rptr).offset +
                       VamanaNode::HEADER_UNTIL_LOCK,
                     &unlock, 1, 0);
}

vec<RemotePtr> MemoryNode::beam_search_candidates(const span<const element_t> query,
                                                  RemotePtr medoid,
                                                  const Configuration& config,
                                                  InsertBreakdownCounters* breakdown) {
  hashset_t<RemotePtr> visited;
  vec<BeamEntry> beam;

  auto t_snapshot = std::chrono::steady_clock::now();
  NodeSnapshot medoid_snapshot;
  read_node_snapshot(medoid, medoid_snapshot);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
  }
  auto t_distance = std::chrono::steady_clock::now();
  const distance_t medoid_dist = distance_to_stored_vector(query, medoid_snapshot.vector_data.data(), config);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
  }

  beam.push_back({medoid, medoid_dist, false});
  visited.insert(medoid);

#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
  // DEBUG: per-insert shard locality summary
  static std::atomic<u32> insert_seq{0};
  u32 this_insert = insert_seq.fetch_add(1, std::memory_order_relaxed);
  bool should_log = (this_insert < 5) || (this_insert % 500 == 0);
  u32 iter_count = 0;
  u32 local_unvisited_sum = 0, remote_unvisited_sum = 0;
#endif

  for (;;) {
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
    ++iter_count;
#endif
    i32 best_idx = -1;
    distance_t best_dist = std::numeric_limits<distance_t>::max();
    auto t_select = std::chrono::steady_clock::now();
    for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
      if (!beam[i].expanded && beam[i].distance < best_dist) {
        best_dist = beam[i].distance;
        best_idx = i;
      }
    }
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_select_ns += elapsed_ns_since(t_select);
    }
    if (best_idx < 0) {
      break;
    }

    beam[best_idx].expanded = true;
    auto t_neighbor_read = std::chrono::steady_clock::now();
    const vec<RemotePtr> neighbors = read_neighbor_list(beam[best_idx].rptr);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_neighbor_read_ns += elapsed_ns_since(t_neighbor_read);
    }
    vec<RemotePtr> unvisited_neighbors;
    unvisited_neighbors.reserve(neighbors.size());
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
    u32 iter_local = 0, iter_remote = 0;
#endif
    for (const RemotePtr& neighbor : neighbors) {
      if (neighbor.is_null() || visited.contains(neighbor)) {
        continue;
      }
      visited.insert(neighbor);
      unvisited_neighbors.push_back(neighbor);
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
      if (local_shard(neighbor.memory_node())) ++iter_local; else ++iter_remote;
#endif
    }
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
    local_unvisited_sum += iter_local;
    remote_unvisited_sum += iter_remote;
#endif

    const u32 snapshot_batch = storage_owner_snapshot_batch_size(config, current_storage_owner_thread_);
    const u32 construction_width = storage_owner_construction_width(config);
    for (size_t begin = 0; begin < unvisited_neighbors.size(); begin += snapshot_batch) {
      const size_t end = std::min(unvisited_neighbors.size(), begin + snapshot_batch);
      vec<RemotePtr> batch;
      batch.reserve(end - begin);
      batch.insert(batch.end(), unvisited_neighbors.begin() + begin, unvisited_neighbors.begin() + end);
      t_snapshot = std::chrono::steady_clock::now();
      vec<NodeSnapshot> snapshots = read_node_snapshots_batched(batch, config);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
      }
      for (const NodeSnapshot& snapshot : snapshots) {
        if (snapshot.deleted) {
          continue;
        }
        t_distance = std::chrono::steady_clock::now();
        const distance_t dist = distance_to_stored_vector(query, snapshot.vector_data.data(), config);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
        }
        auto t_beam_update = std::chrono::steady_clock::now();
        insert_into_beam(beam, snapshot.rptr, dist, construction_width);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_beam_update_ns += elapsed_ns_since(t_beam_update);
        }
      }
    }
  }

#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
  // DEBUG: per-insert summary
  if (should_log) {
    u32 total = local_unvisited_sum + remote_unvisited_sum;
    float local_pct = total > 0 ? 100.0f * local_unvisited_sum / total : 0;
    std::cerr << "[beam_search] insert=" << this_insert
              << " shard=" << storage_id_
              << " iters=" << iter_count
              << " local=" << local_unvisited_sum
              << " remote=" << remote_unvisited_sum
              << " local_pct=" << local_pct << "%"
              << std::endl;
  }
#endif

  vec<RemotePtr> candidates;
  candidates.reserve(beam.size());
  auto t_sort = std::chrono::steady_clock::now();
  std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) {
    return lhs.distance < rhs.distance;
  });
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_result_sort_ns += elapsed_ns_since(t_sort);
  }
  for (const auto& entry : beam) {
    candidates.push_back(entry.rptr);
  }
  return candidates;
}

auto MemoryNode::beam_search_candidates_async(const span<const element_t> query,
                                              RemotePtr medoid,
                                              const Configuration& config,
                                              StorageOwnerThread& thread,
                                              InsertBreakdownCounters* breakdown) -> StorageOwnerInsertCoroutine {
  hashset_t<RemotePtr> visited;
  vec<BeamEntry> beam;

  auto t_snapshot = std::chrono::steady_clock::now();
  NodeSnapshot medoid_snapshot = co_await async_read_node_snapshot(medoid, thread);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
  }
  auto t_distance = std::chrono::steady_clock::now();
  const distance_t medoid_dist = distance_to_stored_vector(query, medoid_snapshot.vector_data.data(), config);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
  }

  beam.push_back({medoid, medoid_dist, false});
  visited.insert(medoid);

#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
  // DEBUG: per-insert per-shard distribution
  static std::atomic<u32> async_insert_seq{0};
  u32 this_insert_a = async_insert_seq.fetch_add(1, std::memory_order_relaxed);
  bool should_log_a = (this_insert_a < 5) || (this_insert_a % 500 == 0);
  u32 iter_count_a = 0;
  u32 shard_hist[6] = {0};  // [0..3]=remote by shard, [4]=local(self), [5]=total expanded
#endif

  for (;;) {
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
    ++iter_count_a;
#endif
    i32 best_idx = -1;
    distance_t best_dist = std::numeric_limits<distance_t>::max();
    auto t_select = std::chrono::steady_clock::now();
    for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
      if (!beam[i].expanded && beam[i].distance < best_dist) {
        best_dist = beam[i].distance;
        best_idx = i;
      }
    }
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_select_ns += elapsed_ns_since(t_select);
    }
    if (best_idx < 0) {
      break;
    }

    beam[best_idx].expanded = true;
    auto t_neighbor_read = std::chrono::steady_clock::now();
    const vec<RemotePtr> neighbors = co_await async_read_neighbor_list(beam[best_idx].rptr, thread);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_neighbor_read_ns += elapsed_ns_since(t_neighbor_read);
    }
    vec<RemotePtr> unvisited_neighbors;
    unvisited_neighbors.reserve(neighbors.size());
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
    u32 expanded_shard = beam[best_idx].rptr.memory_node();
#endif
    for (const RemotePtr& neighbor : neighbors) {
      if (neighbor.is_null() || visited.contains(neighbor)) {
        continue;
      }
      visited.insert(neighbor);
      unvisited_neighbors.push_back(neighbor);
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
      u32 ns = neighbor.memory_node();
      if (ns == storage_id_) ++shard_hist[4]; else ++shard_hist[ns];
#endif
    }
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
    ++shard_hist[5];
#endif

    const u32 snapshot_batch = storage_owner_snapshot_batch_size(config, &thread);
    const u32 construction_width = storage_owner_construction_width(config);
    for (size_t begin = 0; begin < unvisited_neighbors.size(); begin += snapshot_batch) {
      const size_t end = std::min(unvisited_neighbors.size(), begin + snapshot_batch);
      vec<RemotePtr> batch;
      batch.reserve(end - begin);
      batch.insert(batch.end(), unvisited_neighbors.begin() + begin, unvisited_neighbors.begin() + end);
      t_snapshot = std::chrono::steady_clock::now();
      vec<NodeSnapshot> snapshots = co_await async_read_node_snapshots(batch, config, thread);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
      }
      for (const NodeSnapshot& snapshot : snapshots) {
        if (snapshot.deleted) {
          continue;
        }
        t_distance = std::chrono::steady_clock::now();
        const distance_t dist = distance_to_stored_vector(query, snapshot.vector_data.data(), config);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
        }
        auto t_beam_update = std::chrono::steady_clock::now();
        insert_into_beam(beam, snapshot.rptr, dist, construction_width);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_beam_update_ns += elapsed_ns_since(t_beam_update);
        }
      }
    }
  }

#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
  // DEBUG: per-insert per-shard summary
  if (should_log_a) {
    u32 total_neighbors = 0;
    for (u32 s = 0; s < 5; ++s) total_neighbors += shard_hist[s];
    std::cerr << "[beam_search_async] insert=" << this_insert_a
              << " self=" << storage_id_
              << " iters=" << iter_count_a
              << " expanded=" << shard_hist[5]
              << " local=" << shard_hist[4];
    for (u32 s = 0; s < 5; ++s) {
      if (s == storage_id_) continue;
      float pct = total_neighbors > 0 ? 100.0f * shard_hist[s] / total_neighbors : 0;
      std::cerr << " sh" << s << "=" << shard_hist[s] << "(" << int(pct) << "%)";
    }
    std::cerr << std::endl;
  }
#endif

  auto& out = storage_owner_async_candidates_[thread.id][thread.running_coroutine];
  out.clear();
  out.reserve(beam.size());
  auto t_sort = std::chrono::steady_clock::now();
  std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) {
    return lhs.distance < rhs.distance;
  });
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_result_sort_ns += elapsed_ns_since(t_sort);
  }
  for (const auto& entry : beam) {
    out.push_back(entry.rptr);
  }
}

vec<RemotePtr> MemoryNode::robust_prune_cpu(const byte_t* source,
                                            VectorDType source_dtype,
                                            const vec<RemotePtr>& candidates,
                                            const hashset_t<RemotePtr>& skip,
                                            const Configuration& config,
                                            InsertBreakdownCounters* breakdown) {
  struct CandidateInfo {
    RemotePtr rptr;
    distance_t dist{};
    vec<byte_t> vector_data;
  };

  vec<CandidateInfo> infos;
  infos.reserve(candidates.size());
  vec<RemotePtr> filtered;
  filtered.reserve(std::min<size_t>(candidates.size(), storage_owner_prune_candidate_limit(config)));
  const u32 prune_candidate_limit = storage_owner_prune_candidate_limit(config);
  for (const RemotePtr& candidate : candidates) {
    if (candidate.is_null() || skip.contains(candidate)) {
      continue;
    }
    filtered.push_back(candidate);
    if (filtered.size() >= prune_candidate_limit) {
      break;
    }
  }

  const u32 snapshot_batch = storage_owner_snapshot_batch_size(config, current_storage_owner_thread_);
  for (size_t begin = 0; begin < filtered.size(); begin += snapshot_batch) {
    const size_t end = std::min(filtered.size(), begin + snapshot_batch);
    vec<RemotePtr> batch;
    batch.reserve(end - begin);
    batch.insert(batch.end(), filtered.begin() + begin, filtered.begin() + end);
    auto t_snapshot = std::chrono::steady_clock::now();
    vec<NodeSnapshot> snapshots = read_node_snapshots_batched(batch, config);
    if (breakdown != nullptr) {
      breakdown->storage_owner_prune_snapshot_read_ns += elapsed_ns_since(t_snapshot);
    }
    for (NodeSnapshot& snapshot : snapshots) {
      if (snapshot.deleted) {
        continue;
      }
      auto t_distance = std::chrono::steady_clock::now();
      const distance_t dist = distance_between_vectors(source, source_dtype,
                                                       snapshot.vector_data.data(), VamanaNode::vector_dtype(), config);
      if (breakdown != nullptr) {
        breakdown->storage_owner_prune_distance_ns += elapsed_ns_since(t_distance);
      }
      infos.push_back({snapshot.rptr, dist, std::move(snapshot.vector_data)});
    }
  }

  auto t_sort = std::chrono::steady_clock::now();
  std::sort(infos.begin(), infos.end(), [](const CandidateInfo& lhs, const CandidateInfo& rhs) {
    return lhs.dist < rhs.dist;
  });
  if (breakdown != nullptr) {
    breakdown->storage_owner_prune_sort_ns += elapsed_ns_since(t_sort);
  }

  vec<RemotePtr> selected;
  selected.reserve(config.R);
  vec<const byte_t*> selected_vectors;
  selected_vectors.reserve(config.R);

  for (const auto& candidate : infos) {
    if (selected.size() >= config.R) {
      break;
    }

    bool pruned = false;
    for (idx_t i = 0; i < selected_vectors.size(); ++i) {
      auto t_pair_distance = std::chrono::steady_clock::now();
      const distance_t pair_dist = distance_between_vectors(candidate.vector_data.data(), VamanaNode::vector_dtype(),
                                                           selected_vectors[i], VamanaNode::vector_dtype(), config);
      if (breakdown != nullptr) {
        breakdown->storage_owner_prune_pair_distance_ns += elapsed_ns_since(t_pair_distance);
      }
      if (config.alpha * pair_dist <= candidate.dist) {
        pruned = true;
        break;
      }
    }

    if (!pruned) {
      selected.push_back(candidate.rptr);
      selected_vectors.push_back(candidate.vector_data.data());
    }
  }

  return selected;
}

auto MemoryNode::execute_storage_owner_insert_job_async(StorageOwnerThread& thread,
                                            StorageOwnerInsertJob& job,
                                            std::unordered_map<u64, vec<RemotePtr>>& local_updates,
                                            std::unordered_map<u32, vec<service::storage_owner::ReverseUpdateOp>>& remote_updates,
                                            InsertBreakdownCounters& breakdown,
                                            const Configuration& config) -> StorageOwnerInsertCoroutine {
  const auto components = span<const element_t>{reinterpret_cast<const element_t*>(job.vector_data.data()),
                                                 VamanaNode::DIM};
  FreshnessEntry old_entry{};
  u32 generation = 0;
  const auto status = prepare_mutation(job.id, job.kind, &old_entry, &generation);
  job.old_ptr = old_entry.current;
  job.generation = generation;
  if (status != service::storage_owner::MutationStatus::ok) {
    job.status = status;
    job.ok = false;
    co_return;
  }
  if (job.kind == service::storage_owner::MutationKind::erase) {
    job.ok = mark_node_deleted(old_entry.current, old_entry.generation);
    job.status = job.ok ? service::storage_owner::MutationStatus::ok
                        : service::storage_owner::MutationStatus::failed;
    if (job.ok) {
      publish_mutation(job.id, old_entry.current, old_entry.generation, true);
    }
    co_return;
  }
  auto t_medoid = std::chrono::steady_clock::now();
  RemotePtr medoid_ptr = co_await async_read_global_medoid(thread);
  breakdown.storage_owner_medoid_ns += elapsed_ns_since(t_medoid);
  if (medoid_ptr.is_null()) {
    const RemotePtr new_ptr = allocate_local_node();
    job.new_ptr = new_ptr;
    auto t_write = std::chrono::steady_clock::now();
    write_new_node(new_ptr, job.id, components, {}, generation);
    breakdown.storage_owner_write_node_ns += elapsed_ns_since(t_write);
    RemotePtr observed;
    if (try_set_global_medoid(RemotePtr{}, new_ptr, observed) || observed.is_null()) {
      job.ok = true;
      job.status = service::storage_owner::MutationStatus::ok;
      if (job.kind == service::storage_owner::MutationKind::upsert && !old_entry.deleted) {
        mark_node_deleted(old_entry.current, old_entry.generation);
      }
      publish_mutation(job.id, new_ptr, generation, false);
      co_return;
    }
    medoid_ptr = observed;
  }

  auto t_search = std::chrono::steady_clock::now();
  if (config.storage_owner_transitive_search) {
    auto search = beam_search_candidates_transitive_async(
      components, medoid_ptr, config, thread, &breakdown);
    co_await std::suspend_always{};
    while (!search.handle.done()) {
      if (thread.is_ready(thread.running_coroutine)) {
        search.handle.resume();
      } else {
        co_await std::suspend_always{};
      }
    }
    search.handle.destroy();
    breakdown.storage_owner_search_ns += elapsed_ns_since(t_search);
  } else {
    auto search = beam_search_candidates_async(components, medoid_ptr, config, thread, &breakdown);
    co_await std::suspend_always{};
    while (!search.handle.done()) {
      if (thread.is_ready(thread.running_coroutine)) {
        search.handle.resume();
      } else {
        co_await std::suspend_always{};
      }
    }
    search.handle.destroy();
    breakdown.storage_owner_search_ns += elapsed_ns_since(t_search);
  }

  const vec<RemotePtr>& candidates = storage_owner_async_candidates_[thread.id][thread.running_coroutine];
  hashset_t<RemotePtr> empty_skip;
  auto t_prune = std::chrono::steady_clock::now();
  vec<RemotePtr> selected_neighbors = robust_prune_cpu(reinterpret_cast<const byte_t*>(components.data()),
                                                       VectorDType::float32, candidates, empty_skip, config, &breakdown);
  breakdown.storage_owner_prune_ns += elapsed_ns_since(t_prune);
  const RemotePtr new_ptr = allocate_local_node();
  job.new_ptr = new_ptr;
  auto t_write = std::chrono::steady_clock::now();
  write_new_node(new_ptr, job.id, components, selected_neighbors, generation);
  breakdown.storage_owner_write_node_ns += elapsed_ns_since(t_write);
  if (job.kind == service::storage_owner::MutationKind::upsert && !old_entry.deleted) {
    mark_node_deleted(old_entry.current, old_entry.generation);
  }
  publish_mutation(job.id, new_ptr, generation, false);

  for (const RemotePtr& neighbor_ptr : selected_neighbors) {
    if (local_shard(neighbor_ptr.memory_node())) {
      local_updates[neighbor_ptr.raw_address].push_back(new_ptr);
    } else {
      remote_updates[neighbor_ptr.memory_node()].push_back(
        service::storage_owner::ReverseUpdateOp{neighbor_ptr.raw_address, new_ptr.raw_address});
    }
  }
  job.ok = true;
  job.status = service::storage_owner::MutationStatus::ok;
}

bool MemoryNode::apply_local_reverse_update(RemotePtr target_ptr,
                                const vec<RemotePtr>& candidate_ptrs,
                                const Configuration& config) {
  lib_assert(local_shard(target_ptr.memory_node()), "target reverse update must be local");
  if (candidate_ptrs.empty()) {
    return true;
  }

  const auto update_started = std::chrono::steady_clock::now();
  const auto lock_started = std::chrono::steady_clock::now();
  lock_node(target_ptr);
  const u64 lock_wait_ns = elapsed_ns_since(lock_started);
  vec<RemotePtr> updated_neighbors;
  bool changed = false;
  bool pruned = false;
  size_t current_count = 0;
  size_t filtered_count = 0;
  u64 snapshot_ns = 0;
  u64 neighbor_read_ns = 0;
  u64 filter_ns = 0;
  u64 prune_ns = 0;
  u64 write_ns = 0;

  NodeSnapshot target_snapshot;
  auto step_started = std::chrono::steady_clock::now();
  read_node_snapshot(target_ptr, target_snapshot);
  snapshot_ns = elapsed_ns_since(step_started);

  step_started = std::chrono::steady_clock::now();
  vec<RemotePtr> current_neighbors = read_neighbor_list(target_ptr);
  neighbor_read_ns = elapsed_ns_since(step_started);
  current_count = current_neighbors.size();

  step_started = std::chrono::steady_clock::now();
  vec<RemotePtr> filtered_candidates;
  filtered_candidates.reserve(candidate_ptrs.size());
  for (const RemotePtr& candidate_ptr : candidate_ptrs) {
    if (candidate_ptr.is_null()) {
      continue;
    }
    bool already_present = false;
    for (const RemotePtr& existing : current_neighbors) {
      if (existing == candidate_ptr) {
        already_present = true;
        break;
      }
    }
    if (!already_present &&
        std::find(filtered_candidates.begin(), filtered_candidates.end(), candidate_ptr) == filtered_candidates.end()) {
      filtered_candidates.push_back(candidate_ptr);
    }
  }
  filter_ns = elapsed_ns_since(step_started);
  filtered_count = filtered_candidates.size();

  if (!filtered_candidates.empty()) {
    changed = true;

    if (current_neighbors.size() + filtered_candidates.size() <= config.R) {
      current_neighbors.insert(current_neighbors.end(), filtered_candidates.begin(), filtered_candidates.end());
      updated_neighbors = std::move(current_neighbors);
    } else {
      // Evict-farthest: for each new candidate, compute distance from target
      // and replace the farthest existing neighbor if the candidate is closer.
      // This is O(R) distance calls per candidate instead of O(R²) pair distances
      // from full RobustPrune, trading a small diversity loss for large speedup.
      pruned = true;
      step_started = std::chrono::steady_clock::now();

      // 1. Collect non-null current neighbors (do this once, reuse below)
      vec<RemotePtr> non_null_neighbors;
      non_null_neighbors.reserve(current_neighbors.size());
      for (const auto& n : current_neighbors) {
        if (!n.is_null()) non_null_neighbors.push_back(n);
      }

      // 2. Batch-read all current neighbor snapshots + compute distances (O(R), SIMD)
      vec<distance_t> neighbor_dists;
      neighbor_dists.reserve(non_null_neighbors.size());
      if (!non_null_neighbors.empty()) {
        vec<NodeSnapshot> neighbor_snapshots =
            read_node_snapshots_batched(non_null_neighbors, config);
        for (const auto& snap : neighbor_snapshots) {
          neighbor_dists.push_back(distance_between_vectors(
              target_snapshot.vector_data.data(), VamanaNode::vector_dtype(),
              snap.vector_data.data(), VamanaNode::vector_dtype(), config));
        }
      }

      // 3. Initialise updated_neighbors from filtered list (no extra allocation)
      updated_neighbors = std::move(non_null_neighbors);

      // 4. For each candidate, evict farthest if candidate is closer
      {
        vec<RemotePtr> non_null_candidates;
        non_null_candidates.reserve(filtered_candidates.size());
        for (const auto& c : filtered_candidates) {
          if (!c.is_null()) non_null_candidates.push_back(c);
        }

        vec<NodeSnapshot> candidate_snapshots;
        if (!non_null_candidates.empty()) {
          candidate_snapshots = read_node_snapshots_batched(non_null_candidates, config);
        }

        for (size_t ci = 0; ci < candidate_snapshots.size(); ++ci) {
          const auto& cand_snap = candidate_snapshots[ci];
          const distance_t cand_dist = distance_between_vectors(
              target_snapshot.vector_data.data(), VamanaNode::vector_dtype(),
              cand_snap.vector_data.data(), VamanaNode::vector_dtype(), config);

          if (updated_neighbors.size() < config.R) {
            updated_neighbors.push_back(cand_snap.rptr);
            neighbor_dists.push_back(cand_dist);
          } else {
            // updated_neighbors.size() >= R, and neighbor_dists tracks the same
            // set, so at least one element exists.
            lib_assert(!neighbor_dists.empty(),
                       "neighbor_dists non-empty when updated_neighbors >= R");
            size_t farthest_idx = 0;
            distance_t farthest_dist = neighbor_dists[0];
            for (size_t j = 1; j < neighbor_dists.size(); ++j) {
              if (neighbor_dists[j] > farthest_dist) {
                farthest_dist = neighbor_dists[j];
                farthest_idx = j;
              }
            }
            if (cand_dist < farthest_dist) {
              updated_neighbors[farthest_idx] = cand_snap.rptr;
              neighbor_dists[farthest_idx] = cand_dist;
            }
          }
        }
      }

      prune_ns = elapsed_ns_since(step_started);
    }
  }

  if (changed) {
    step_started = std::chrono::steady_clock::now();
    write_neighbor_list(target_ptr, updated_neighbors);
    write_ns = elapsed_ns_since(step_started);
  }
  unlock_node(target_ptr);

  const u64 update_ns = elapsed_ns_since(update_started);
  if (update_ns > 1000ull * 1000ull * 1000ull) {
    static std::atomic<u32> slow_update_logs{0};
    const u32 log_index = slow_update_logs.fetch_add(1, std::memory_order_relaxed);
    if (log_index < 16) {
      std::cerr << "[storage-owner] slow reverse-update target"
                << " self_shard=" << storage_id_
                << " target_raw=" << target_ptr.raw_address
                << " candidates=" << candidate_ptrs.size()
                << " current_neighbors=" << current_count
                << " filtered_candidates=" << filtered_count
                << " changed=" << (changed ? 1 : 0)
                << " pruned=" << (pruned ? 1 : 0)
                << " elapsed_ms=" << (update_ns / 1000000.0)
                << " lock_wait_ms=" << (lock_wait_ns / 1000000.0)
                << " snapshot_ms=" << (snapshot_ns / 1000000.0)
                << " neighbor_read_ms=" << (neighbor_read_ns / 1000000.0)
                << " filter_ms=" << (filter_ns / 1000000.0)
                << " prune_ms=" << (prune_ns / 1000000.0)
                << " write_ms=" << (write_ns / 1000000.0)
                << std::endl;
    }
  }
  return true;
}

// ─── Transitive (handoff-based) beam search ──────────────────────────

// Check whether a RemotePtr points within the local shard bounds.
// Returns false for garbage pointers that would crash read_node_snapshot / read_neighbor_list.
inline bool ptr_in_bounds(RemotePtr rptr, u64 shard_cap) {
  return vamana::StorageLayoutResolver::ptr_in_bounds(rptr, shard_cap);
}

// Expand ALL unexpanded local nodes in the beam.
// Remote nodes are deferred (only added to beam with approximate distance).
bool MemoryNode::expand_all_local_nodes(vec<BeamEntry>& beam,
                                            hashset_t<RemotePtr>& visited,
                                            const span<const element_t> query,
                                            const Configuration& config,
                                            u32 local_shard_id,
                                            InsertBreakdownCounters* breakdown,
                                            u64* expanded_count,
                                            u64* snapshot_read_count,
                                            u64* neighbor_read_count) {
  bool any_expanded = false;
  const u32 construction_width = storage_owner_construction_width(config);
  const u64 shard_cap = mn_memory_bytes_;

  for (;;) {
    // Find best unexpanded LOCAL node
    i32 best_idx = -1;
    distance_t best_dist = std::numeric_limits<distance_t>::max();
    auto t_select = std::chrono::steady_clock::now();
    for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
      if (!beam[i].expanded && beam[i].distance < best_dist) {
        best_dist = beam[i].distance;
        best_idx = i;
      }
    }
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_select_ns += elapsed_ns_since(t_select);
    }
    if (best_idx < 0) break;

    const RemotePtr best_rptr = beam[best_idx].rptr;
    // Only expand local nodes; break if the best candidate is remote
    if (best_rptr.memory_node() != local_shard_id) break;

    beam[best_idx].expanded = true;
    any_expanded = true;
    if (expanded_count != nullptr) {
      ++*expanded_count;
    }

    auto t_nbr = std::chrono::steady_clock::now();
    const vec<RemotePtr> neighbors = ptr_in_bounds(best_rptr, shard_cap)
        ? read_neighbor_list(best_rptr)
        : vec<RemotePtr>{};
    if (neighbor_read_count != nullptr && ptr_in_bounds(best_rptr, shard_cap)) {
      ++*neighbor_read_count;
    }
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_neighbor_read_ns += elapsed_ns_since(t_nbr);
    }

    vec<RemotePtr> local_unvisited;
    vec<RemotePtr> remote_unvisited;
    for (const RemotePtr& neighbor : neighbors) {
      if (neighbor.is_null() || visited.contains(neighbor)) continue;
      visited.insert(neighbor);
      if (neighbor.memory_node() == local_shard_id) {
        local_unvisited.push_back(neighbor);
      } else {
        remote_unvisited.push_back(neighbor);
      }
    }

    // Process local unvisited nodes: read snapshots, compute distances
    const u32 snapshot_batch = storage_owner_snapshot_batch_size(config, current_storage_owner_thread_);
    for (size_t begin = 0; begin < local_unvisited.size(); begin += snapshot_batch) {
      const size_t end = std::min(local_unvisited.size(), begin + snapshot_batch);
      vec<RemotePtr> batch;
      batch.reserve(end - begin);
      for (size_t bi = begin; bi < end; ++bi) {
        if (ptr_in_bounds(local_unvisited[bi], shard_cap)) {
          batch.push_back(local_unvisited[bi]);
        }
      }
      if (batch.empty()) continue;
      auto t_snap = std::chrono::steady_clock::now();
      vec<NodeSnapshot> snapshots = read_node_snapshots_batched(batch, config);
      if (snapshot_read_count != nullptr) {
        *snapshot_read_count += batch.size();
      }
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snap);
      }
      for (const NodeSnapshot& snapshot : snapshots) {
        if (snapshot.deleted) {
          continue;
        }
        auto t_dist = std::chrono::steady_clock::now();
        const distance_t dist = distance_to_stored_vector(query, snapshot.vector_data.data(), config);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_dist);
        }
        auto t_beam = std::chrono::steady_clock::now();
        insert_into_beam(beam, snapshot.rptr, dist, construction_width);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_beam_update_ns += elapsed_ns_since(t_beam);
        }
      }
    }

    // Defer remote nodes: add to beam with approximate distance (parent distance)
    for (const RemotePtr& rptr : remote_unvisited) {
      insert_into_beam(beam, rptr, best_dist, construction_width);
    }
  }

  return any_expanded;
}


// Originator-side transitive beam search (coroutine version).
// Exhausts local nodes, then hands off to remote shards via RPC.
auto MemoryNode::beam_search_candidates_transitive_async(
    const span<const element_t> query,
    RemotePtr medoid,
    const Configuration& config,
    StorageOwnerThread& thread,
    InsertBreakdownCounters* breakdown) -> StorageOwnerInsertCoroutine {

  hashset_t<RemotePtr> visited;
  vec<BeamEntry> beam;

  auto t_snap = std::chrono::steady_clock::now();
  NodeSnapshot medoid_snap = co_await async_read_node_snapshot(medoid, thread);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snap);
  }
  auto t_dist = std::chrono::steady_clock::now();
  const distance_t medoid_dist = distance_to_stored_vector(query, medoid_snap.vector_data.data(), config);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_dist);
  }

  beam.push_back({medoid, medoid_dist, false});
  visited.insert(medoid);

  // Track which shards have been searched (local + handoffs)
  vec<byte_t> shard_searched(config.num_server_nodes(), 0);
  shard_searched[storage_id_] = 1;

  for (;;) {
    // Phase 1: Expand all local nodes
    expand_all_local_nodes(beam, visited, query, config, storage_id_, breakdown);

    // Phase 2: collect every remote shard currently represented in the frontier.
    // Issuing these handoffs together avoids serial shard-by-shard update search.
    vec<HandoffTargetShard> targets = collect_handoff_targets(
      beam, shard_searched, storage_id_, config.num_server_nodes());
    if (targets.empty()) {
      break;
    }

    struct PendingHandoff {
      u32 target_shard{};
      size_t request_bytes{};
      SearchHandoffAwaitable awaitable;
    };
    vec<PendingHandoff> pending;
    pending.reserve(targets.size());
    for (const HandoffTargetShard& target : targets) {
      vec<byte_t> message = build_search_handoff_request(
        query,
        beam,
        visited,
        target.shard,
        storage_id_,
        next_peer_request_id_.fetch_add(1, std::memory_order_relaxed),
        config);
      const size_t request_bytes = message.size();
      pending.push_back(PendingHandoff{
        target.shard,
        request_bytes,
        async_search_handoff(target.shard, std::move(message), thread, config)});
      shard_searched[target.shard] = 1;
    }

    struct CompletedHandoff {
      u32 target_shard{};
      HandoffResult result;
    };
    vec<CompletedHandoff> completed;
    completed.reserve(pending.size());
    bool queue_full = false;
    for (PendingHandoff& item : pending) {
      HandoffResult handoff = co_await item.awaitable;
      record_search_handoff_stats(breakdown, handoff, item.request_bytes);
      if (breakdown != nullptr) {
        breakdown->storage_owner_handoff_queue_wait_ns += handoff.queue_wait_ns;
        breakdown->storage_owner_handoff_send_ns += handoff.send_ns;
        breakdown->storage_owner_handoff_response_wait_ns += handoff.response_wait_ns;
      }
      queue_full = queue_full || handoff.status == HandoffResultStatus::queue_full;
      completed.push_back(CompletedHandoff{item.target_shard, std::move(handoff)});
    }
    if (queue_full) {
      auto fallback = beam_search_candidates_async(query, medoid, config, thread, breakdown);
      co_await std::suspend_always{};
      while (!fallback.handle.done()) {
        if (thread.is_ready(thread.running_coroutine)) {
          fallback.handle.resume();
        } else {
          co_await std::suspend_always{};
        }
      }
      fallback.handle.destroy();
      co_return;
    }

    const u32 construction_width = storage_owner_construction_width(config);
    for (const CompletedHandoff& item : completed) {
      if (!merge_search_handoff_response(beam, visited, item.result, construction_width)) {
        mark_shard_frontier_expanded(beam, item.target_shard);
      }
    }
  }

  // Build result from beam
  auto& out = storage_owner_async_candidates_[thread.id][thread.running_coroutine];
  out.clear();
  out.reserve(beam.size());
  auto t_sort = std::chrono::steady_clock::now();
  std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) {
    return lhs.distance < rhs.distance;
  });
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_result_sort_ns += elapsed_ns_since(t_sort);
  }
  for (const auto& entry : beam) {
    out.push_back(entry.rptr);
  }
  co_return;
}
