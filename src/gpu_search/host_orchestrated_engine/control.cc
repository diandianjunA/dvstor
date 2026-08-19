#include "gpu_search/host_orchestrated_engine/impl.hh"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "gpu_search/centroid_route_poll_policy.hh"
#include "gpu_search/maintenance_fence.hh"

namespace gpu_search {
namespace {

size_t control_stride() {
  const size_t bytes = std::max(sizeof(format::StorageControlBlock),
                                sizeof(maintenance_telemetry::Snapshot));
  return (bytes + 63u) & ~size_t{63u};
}

bool validate_route_probe_control(
    const format::StorageCentroidRoutePublicationHeader& header,
    const format::StorageCentroidRouteDescriptor& descriptor,
    u32 shard, std::string& error) {
  const auto scalar_type = static_cast<format::CentroidScalarType>(
    header.centroid_scalar_type);
  const u32 scalar_bytes = format::centroid_scalar_bytes(scalar_type);
  const u64 expected_bytes = format::storage_centroid_route_publication_bytes(
    header.dim, scalar_type, header.live_entry_capacity);
  const u64 expected_entries_offset = format::align_up(
    sizeof(format::StorageCentroidRoutePublicationHeader) +
      static_cast<u64>(header.dim) * scalar_bytes,
    alignof(format::StorageCentroidRouteEntry));
  if (header.magic != format::kStorageCentroidRoutePublicationMagic ||
      header.version != format::kStorageCentroidRoutePublicationVersion ||
      header.header_bytes !=
        sizeof(format::StorageCentroidRoutePublicationHeader) ||
      header.total_bytes != descriptor.publication_bytes ||
      header.shard_id != shard || header.dim != descriptor.dim ||
      header.centroid_scalar_type != descriptor.centroid_scalar_type ||
      header.live_entry_capacity != descriptor.live_entry_capacity ||
      header.live_entry_count > header.live_entry_capacity ||
      header.reserved0 != 0 || header.reserved[0] != 0 ||
      header.reserved[1] != 0 || header.shard_version == 0 ||
      expected_bytes != descriptor.publication_bytes ||
      header.centroid_offset !=
        sizeof(format::StorageCentroidRoutePublicationHeader) ||
      header.centroid_bytes != static_cast<u64>(header.dim) * scalar_bytes ||
      header.entries_offset != expected_entries_offset ||
      header.entries_bytes != static_cast<u64>(header.live_entry_capacity) *
        sizeof(format::StorageCentroidRouteEntry) ||
      (header.vector_count == 0) != (header.live_entry_count == 0)) {
    error = "storage centroid route probe header mismatch";
    return false;
  }
  return true;
}

centroid_route_poll::PublicationIdentity identity(
    const format::StorageCentroidRoutePublicationHeader& header) {
  return {
    .sequence = header.sequence,
    .version = header.shard_version,
    .body_checksum = header.body_checksum,
    .vector_count = header.vector_count,
    .live_entry_count = header.live_entry_count,
  };
}

}  // namespace

void HostOrchestratedSearchEngine::Impl::validate_storage_control(
    const format::StorageControlBlock& control, size_t shard) const {
  std::string route_error;
  bool valid_route = format::validate_storage_centroid_route_descriptor(
    control.centroid_route, index.layout.dim,
    static_cast<u32>(index.shards.size()), &route_error);
  if (valid_route && control.centroid_route.centroid_scalar_type !=
      static_cast<u32>(format::CentroidScalarType::float32)) {
    valid_route = false;
    route_error = "centroid routing is not canonical float32";
  }
  if (control.magic != format::kStorageControlMagic ||
      control.version != format::kStorageControlVersion ||
      control.header_bytes != sizeof(format::StorageControlBlock) ||
      control.shard_id != shard ||
      control.dynamic_record_bytes != index.shards[shard].dynamic_record_bytes ||
      control.dynamic_hot_offset != index.shards[shard].dynamic_hot_offset ||
      control.dynamic_code_offset != index.shards[shard].dynamic_code_offset ||
      control.code_bytes != index.layout.code_bytes || !valid_route) {
    std::ostringstream message;
    message << "storage maintenance control mismatch for host query shard "
            << shard << ": " << route_error;
    throw std::runtime_error(message.str());
  }
}

std::vector<format::StorageControlBlock>
HostOrchestratedSearchEngine::Impl::read_storage_controls(Lane& lane) {
  std::vector<ReadRequest> requests(index.shards.size());
  for (size_t shard = 0; shard < index.shards.size(); ++shard) {
    requests[shard] = {
      .shard = static_cast<u32>(shard),
      .remote_offset = index.shards[shard].control_remote_offset,
      .local_offset = control_scratch_offset + shard * control_stride(),
      .bytes = sizeof(format::StorageControlBlock),
    };
  }
  read_batch(lane, requests, false);
  std::vector<format::StorageControlBlock> controls(index.shards.size());
  for (size_t shard = 0; shard < controls.size(); ++shard) {
    std::memcpy(&controls[shard],
                lane.scratch + control_scratch_offset +
                  shard * control_stride(),
                sizeof(controls[shard]));
    validate_storage_control(controls[shard], shard);
  }
  return controls;
}

void HostOrchestratedSearchEngine::Impl::initialize_storage_routes(
    Lane& lane) {
  const auto controls = read_storage_controls(lane);
  route_descriptors.resize(controls.size());
  route_cache.resize(controls.size());
  for (size_t shard = 0; shard < controls.size(); ++shard) {
    route_descriptors[shard] = controls[shard].centroid_route;
    if (route_descriptors[shard].publication_bytes > route_snapshot_stride ||
        route_descriptors[shard].remote_offset > storage_region_bytes ||
        route_descriptors[shard].publication_bytes >
          storage_region_bytes - route_descriptors[shard].remote_offset) {
      throw std::runtime_error(
        "host centroid route publication exceeds its registered shard");
    }
  }
  if (!synchronize_storage_routes(lane)) {
    throw std::runtime_error(
      "initial host centroid route snapshot was not stable");
  }
}

bool HostOrchestratedSearchEngine::Impl::synchronize_storage_routes(
    Lane& lane) {
  using Header = format::StorageCentroidRoutePublicationHeader;
  const size_t shard_count = index.shards.size();
  std::vector<ReadRequest> probes(shard_count);
  for (size_t shard = 0; shard < shard_count; ++shard) {
    probes[shard] = {
      .shard = static_cast<u32>(shard),
      .remote_offset = route_descriptors[shard].remote_offset,
      .local_offset = route_scratch_offset + shard * sizeof(Header),
      .bytes = sizeof(Header),
    };
  }
  read_batch(lane, probes, false);
  engine.telemetry_.centroid_route_probe_reads.fetch_add(
    shard_count, std::memory_order_relaxed);
  std::vector<Header> headers(shard_count);
  std::memcpy(headers.data(), lane.scratch + route_scratch_offset,
              shard_count * sizeof(Header));

  std::vector<u32> changed;
  changed.reserve(shard_count);
  for (u32 shard = 0; shard < shard_count; ++shard) {
    const Header& header = headers[shard];
    if (header.sequence == 0 || (header.sequence & 1u) != 0) {
      engine.telemetry_.centroid_route_snapshot_skips.fetch_add(
        1, std::memory_order_relaxed);
      return false;
    }
    std::string error;
    if (!validate_route_probe_control(
          header, route_descriptors[shard], shard, error)) {
      if (route_cache[shard].version != 0) {
        engine.telemetry_.centroid_route_snapshot_skips.fetch_add(
          1, std::memory_order_relaxed);
        return false;
      }
      throw std::runtime_error(
        "shard " + std::to_string(shard) + ": " + error);
    }
    const centroid_route_poll::PublicationIdentity cached_identity{
      .sequence = route_cache[shard].publication_sequence,
      .version = route_cache[shard].version,
      .body_checksum = route_cache[shard].body_checksum,
      .vector_count = route_cache[shard].vector_count,
      .live_entry_count = route_cache[shard].live_entry_count,
    };
    const auto action = centroid_route_poll::classify_probe(
      route_cache[shard].version != 0, cached_identity, identity(header));
    if (action == centroid_route_poll::ProbeAction::retry) {
      engine.telemetry_.centroid_route_snapshot_skips.fetch_add(
        1, std::memory_order_relaxed);
      return false;
    }
    if (action == centroid_route_poll::ProbeAction::read_body) {
      changed.push_back(shard);
    }
  }

  if (changed.empty()) {
    engine.telemetry_.centroid_route_unchanged_polls.fetch_add(
      1, std::memory_order_relaxed);
    return false;
  }

  std::vector<ReadRequest> bodies(changed.size());
  std::vector<ReadRequest> sequences(changed.size());
  for (size_t update = 0; update < changed.size(); ++update) {
    const u32 shard = changed[update];
    bodies[update] = {
      .shard = shard,
      .remote_offset = route_descriptors[shard].remote_offset,
      .local_offset = route_scratch_offset + update * route_snapshot_stride,
      .bytes = static_cast<u32>(route_descriptors[shard].publication_bytes),
    };
    sequences[update] = {
      .shard = shard,
      .remote_offset = route_descriptors[shard].remote_offset +
        offsetof(Header, sequence),
      .local_offset = route_sequence_scratch_offset + update * sizeof(u64),
      .bytes = sizeof(u64),
    };
  }
  read_batch(lane, bodies, false);
  read_batch(lane, sequences, false);
  engine.telemetry_.centroid_route_body_reads.fetch_add(
    changed.size(), std::memory_order_relaxed);

  std::vector<RouteShard> next_cache = route_cache;
  for (size_t update = 0; update < changed.size(); ++update) {
    const u32 shard = changed[update];
    const span<const byte_t> publication{
      lane.scratch + route_scratch_offset + update * route_snapshot_stride,
      static_cast<size_t>(route_descriptors[shard].publication_bytes)};
    const auto* body_header = reinterpret_cast<const Header*>(
      publication.data());
    u64 sequence_after = 0;
    std::memcpy(&sequence_after,
                lane.scratch + route_sequence_scratch_offset +
                  update * sizeof(u64),
                sizeof(sequence_after));
    if (!centroid_route_poll::body_read_is_stable(
          identity(headers[shard]), identity(*body_header),
          sequence_after)) {
      engine.telemetry_.centroid_route_snapshot_skips.fetch_add(
        1, std::memory_order_relaxed);
      return false;
    }
    std::string error;
    if (!format::validate_storage_centroid_route_publication(
          publication, route_descriptors[shard], shard, &error)) {
      if (error == "storage centroid route snapshot overlaps publication" ||
          error == "storage centroid route publication checksum mismatch") {
        engine.telemetry_.centroid_route_snapshot_skips.fetch_add(
          1, std::memory_order_relaxed);
        return false;
      }
      throw std::runtime_error(
        "shard " + std::to_string(shard) + ": " + error);
    }
    RouteShard parsed;
    parsed.publication_sequence = body_header->sequence;
    parsed.body_checksum = body_header->body_checksum;
    parsed.version = body_header->shard_version;
    parsed.vector_count = body_header->vector_count;
    parsed.live_entry_count = body_header->live_entry_count;
    parsed.centroid.resize(config.dim);
    const auto* centroid = static_cast<const f32*>(
      format::storage_centroid_route_centroid_data(publication));
    std::copy_n(centroid, config.dim, parsed.centroid.begin());
    const auto entries = format::storage_centroid_route_entries(publication);
    for (u32 entry = 0; entry < parsed.live_entry_count; ++entry) {
      if (entries[entry].remote_node == 0 ||
          entries[entry].flags != format::kStorageCentroidRouteLive ||
          RemotePtr{entries[entry].remote_node}.memory_node() != shard) {
        throw std::runtime_error("invalid host centroid route live entry");
      }
      parsed.entries[entry] = entries[entry];
    }
    if (next_cache[shard].version != 0 &&
        parsed.version < next_cache[shard].version) {
      throw std::runtime_error("storage centroid route version regressed");
    }
    if (next_cache[shard].version == parsed.version &&
        next_cache[shard].version != 0 &&
        next_cache[shard].body_checksum != parsed.body_checksum) {
      throw std::runtime_error(
        "storage centroid route changed without version advance");
    }
    next_cache[shard] = std::move(parsed);
  }

  auto snapshot = std::make_shared<RouteSnapshot>();
  snapshot->shards = next_cache;
  snapshot->home.resize(shard_count);
  u64 live_entries = 0;
  for (u32 shard = 0; shard < shard_count; ++shard) {
    if (next_cache[shard].version == 0 ||
        next_cache[shard].centroid.size() != config.dim) {
      engine.telemetry_.centroid_route_snapshot_skips.fetch_add(
        1, std::memory_order_relaxed);
      return false;
    }
    snapshot->home[shard] = {
      .vector_count = next_cache[shard].vector_count,
      .centroid = next_cache[shard].centroid,
      .live_entry_count = next_cache[shard].live_entry_count,
    };
    live_entries += next_cache[shard].live_entry_count;
  }
  if (live_entries == 0) {
    throw std::runtime_error("storage centroid routes contain no live entry");
  }
  route_cache = std::move(next_cache);
  std::atomic_store_explicit(
    &route_snapshot,
    std::shared_ptr<const RouteSnapshot>(std::move(snapshot)),
    std::memory_order_release);
  engine.telemetry_.centroid_route_live_entries.store(
    live_entries, std::memory_order_relaxed);
  engine.telemetry_.centroid_route_publications.fetch_add(
    1, std::memory_order_relaxed);
  engine.telemetry_.centroid_route_shard_updates.fetch_add(
    changed.size(), std::memory_order_relaxed);
  return true;
}

void HostOrchestratedSearchEngine::Impl::maintenance_loop() {
  centroid_route_poll::AdaptiveIdleBackoff backoff{route_poll_salt};
  while (!maintenance_shutdown.load(std::memory_order_acquire)) {
    {
      std::unique_lock lock(maintenance_mutex);
      maintenance_cv.wait_for(lock, backoff.delay(), [&] {
        return maintenance_shutdown.load(std::memory_order_acquire);
      });
    }
    if (maintenance_shutdown.load(std::memory_order_acquire)) break;
    try {
      LaneGuard lane = acquire_lane();
      std::lock_guard refresh_lock(route_refresh_mutex);
      const bool changed = synchronize_storage_routes(lane.get());
      backoff.observe(changed);
      engine.telemetry_.centroid_route_poll_delay_us.store(
        std::chrono::duration_cast<std::chrono::microseconds>(
          backoff.delay()).count(), std::memory_order_relaxed);
    } catch (const std::exception& error) {
      if (!maintenance_shutdown.load(std::memory_order_acquire)) {
        std::cerr << "[gpu-search] host route refresh failed: "
                  << error.what() << '\n';
      }
      backoff.observe(true);
    }
  }
}

std::optional<u32>
HostOrchestratedSearchEngine::Impl::select_centroid_home(
    std::span<const f32> vector) const {
  const auto snapshot = std::atomic_load_explicit(
    &route_snapshot, std::memory_order_acquire);
  if (snapshot == nullptr) return std::nullopt;
  return centroid_home::select_published_snapshot(vector, snapshot->home);
}

std::vector<std::optional<maintenance_telemetry::Snapshot>>
HostOrchestratedSearchEngine::Impl::read_maintenance_telemetry(Lane& lane) {
  const size_t shard_count = index.shards.size();
  std::vector<std::optional<maintenance_telemetry::Snapshot>> result(
    shard_count);
  std::vector<ReadRequest> reads(shard_count);
  std::vector<maintenance_telemetry::Snapshot> snapshots(shard_count);
  for (u32 attempt = 0; attempt < 3; ++attempt) {
    for (size_t shard = 0; shard < shard_count; ++shard) {
      reads[shard] = {
        .shard = static_cast<u32>(shard),
        .remote_offset = index.shards[shard].control_remote_offset +
          maintenance_telemetry::kSnapshotOffset,
        .local_offset = control_scratch_offset + shard * control_stride(),
        .bytes = sizeof(maintenance_telemetry::Snapshot),
      };
    }
    read_batch(lane, reads, false);
    for (size_t shard = 0; shard < shard_count; ++shard) {
      std::memcpy(&snapshots[shard],
                  lane.scratch + control_scratch_offset +
                    shard * control_stride(),
                  sizeof(snapshots[shard]));
      reads[shard].bytes = sizeof(u64);
    }
    read_batch(lane, reads, false);
    bool retry = false;
    for (size_t shard = 0; shard < shard_count; ++shard) {
      u64 after = 0;
      std::memcpy(&after,
                  lane.scratch + control_scratch_offset +
                    shard * control_stride(),
                  sizeof(after));
      if (maintenance_telemetry::validate(
            snapshots[shard], after, static_cast<u32>(shard))) {
        result[shard] = snapshots[shard];
      } else if (snapshots[shard].sequence != 0 &&
                 (snapshots[shard].magic == maintenance_telemetry::kMagic ||
                  (snapshots[shard].sequence & 1u) != 0)) {
        retry = true;
      }
    }
    if (!retry) break;
  }
  return result;
}

std::vector<std::optional<maintenance_telemetry::Snapshot>>
HostOrchestratedSearchEngine::Impl::read_maintenance_telemetry() {
  LaneGuard lane = acquire_lane();
  return read_maintenance_telemetry(lane.get());
}

bool HostOrchestratedSearchEngine::Impl::wait_for_maintenance(
    std::span<const u64> target_sequences,
    std::chrono::milliseconds timeout,
    std::vector<u64>* durable_sequences,
    std::vector<u64>* effective_target_sequences) {
  if (target_sequences.size() != index.shards.size()) {
    throw std::invalid_argument(
      "maintenance target count does not match storage shard count");
  }
  auto read_controls = [&]() {
    LaneGuard lane = acquire_lane();
    return read_storage_controls(lane.get());
  };
  auto controls = read_controls();
  std::vector<u64> next(controls.size());
  for (size_t shard = 0; shard < controls.size(); ++shard) {
    next[shard] = controls[shard].next_maintenance_sequence;
  }
  const auto targets = maintenance_fence::capture_targets(
    target_sequences, next);
  if (effective_target_sequences != nullptr) {
    *effective_target_sequences = targets;
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  auto delay = std::chrono::milliseconds(1);
  for (;;) {
    bool complete = true;
    std::vector<u64> observed(controls.size());
    for (size_t shard = 0; shard < controls.size(); ++shard) {
      observed[shard] = controls[shard].durable_maintenance_sequence;
      complete = complete && observed[shard] >= targets[shard];
    }
    if (durable_sequences != nullptr) *durable_sequences = observed;
    if (complete) return true;
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) return false;
    std::unique_lock lock(maintenance_mutex);
    maintenance_cv.wait_for(lock, std::min(
      delay, std::chrono::duration_cast<std::chrono::milliseconds>(
               deadline - now)), [&] {
      return maintenance_shutdown.load(std::memory_order_acquire);
    });
    if (maintenance_shutdown.load(std::memory_order_acquire)) return false;
    lock.unlock();
    controls = read_controls();
    delay = std::min(delay * 2, std::chrono::milliseconds(16));
  }
}

}  // namespace gpu_search
