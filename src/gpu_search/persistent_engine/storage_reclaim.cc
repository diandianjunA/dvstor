#include "gpu_search/persistent_engine/impl.hh"
#include "gpu_search/persistent_engine/cuda_helpers.hh"

namespace gpu_search {

using namespace persistent_engine_detail;

namespace {

bool validate_centroid_route_probe(
    const format::StorageCentroidRoutePublicationHeader& header,
    const format::StorageCentroidRouteDescriptor& descriptor,
    u32 expected_shard, std::string& error) {
  const auto scalar_type = static_cast<format::CentroidScalarType>(
    header.centroid_scalar_type);
  const u32 scalar_bytes = format::centroid_scalar_bytes(scalar_type);
  const u64 expected_bytes = format::storage_centroid_route_publication_bytes(
    header.dim, scalar_type, header.live_entry_capacity);
  const u64 expected_entries_offset = align_up(
    sizeof(format::StorageCentroidRoutePublicationHeader) +
      static_cast<u64>(header.dim) * scalar_bytes,
    alignof(format::StorageCentroidRouteEntry));
  if (header.magic != format::kStorageCentroidRoutePublicationMagic ||
      header.version != format::kStorageCentroidRoutePublicationVersion ||
      header.header_bytes !=
        sizeof(format::StorageCentroidRoutePublicationHeader) ||
      header.total_bytes != descriptor.publication_bytes ||
      header.shard_id != expected_shard || header.dim != descriptor.dim ||
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

}  // namespace

void PersistentSearchEngine::Impl::validate_storage_control(const format::StorageControlBlock& control,
                              size_t shard) const {
  std::string centroid_route_error;
  bool valid_centroid_route =
    format::validate_storage_centroid_route_descriptor(
      control.centroid_route, index.layout.dim,
      static_cast<u32>(index.shards.size()), &centroid_route_error);
  if (valid_centroid_route && control.centroid_route.centroid_scalar_type !=
        static_cast<u32>(format::CentroidScalarType::float32)) {
    valid_centroid_route = false;
    centroid_route_error =
      "centroid routing requires the canonical float32 representation";
  }
  if (control.magic != format::kStorageControlMagic ||
      control.version != format::kStorageControlVersion ||
      control.header_bytes != sizeof(format::StorageControlBlock) ||
      control.shard_id != shard ||
      control.dynamic_record_bytes != index.shards[shard].dynamic_record_bytes ||
      control.dynamic_hot_offset != index.shards[shard].dynamic_hot_offset ||
      control.dynamic_code_offset != index.shards[shard].dynamic_code_offset ||
      control.code_bytes != index.layout.code_bytes ||
      !valid_centroid_route) {
    std::ostringstream message;
    message << "storage maintenance control mismatch for shard " << shard
            << ": expected{magic=0x" << std::hex
            << format::kStorageControlMagic << std::dec
            << ",version=" << format::kStorageControlVersion
            << ",header=" << sizeof(format::StorageControlBlock)
            << ",shard=" << shard
            << ",record=" << index.shards[shard].dynamic_record_bytes
            << ",hot=" << index.shards[shard].dynamic_hot_offset
            << ",dynamic_code=" << index.shards[shard].dynamic_code_offset
            << ",code=" << index.layout.code_bytes
            << "} actual{magic=0x" << std::hex << control.magic << std::dec
            << ",version=" << control.version
            << ",header=" << control.header_bytes
            << ",shard=" << control.shard_id
            << ",record=" << control.dynamic_record_bytes
            << ",hot=" << control.dynamic_hot_offset
            << ",dynamic_code=" << control.dynamic_code_offset
            << ",code=" << control.code_bytes
            << ",centroid_route="
            << (valid_centroid_route ? "valid" : centroid_route_error)
            << "}. Rebuild and restart every storage node from the current "
               "dev branch before starting the compute node.";
    throw std::runtime_error(message.str());
  }
}

std::vector<format::StorageControlBlock> PersistentSearchEngine::Impl::read_storage_controls() {
  if (control_bootstrapper == nullptr || index.shards.empty()) return {};
  std::vector<NavigationRead> requests(index.shards.size());
  std::vector<i32> statuses(index.shards.size(), -EIO);
  for (size_t shard = 0; shard < index.shards.size(); ++shard) {
    requests[shard] = NavigationRead{
      .remote_offset = index.shards[shard].control_remote_offset,
      .destination_address = reinterpret_cast<u64>(d_control_snapshots + shard),
      .bytes = sizeof(format::StorageControlBlock),
      .memory_node = static_cast<u16>(shard),
    };
  }
  control_bootstrapper->read(requests, statuses);
  std::vector<format::StorageControlBlock> controls(index.shards.size());
  check_cuda(cudaMemcpy(controls.data(), d_control_snapshots,
                        controls.size() * sizeof(format::StorageControlBlock),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(storage maintenance controls)");
  for (size_t shard = 0; shard < controls.size(); ++shard) {
    if (statuses[shard] <= 0) {
      throw std::runtime_error(
        "storage maintenance control read failed for shard " +
        std::to_string(shard));
    }
    validate_storage_control(controls[shard], shard);
  }
  return controls;
}

PersistentSearchEngine::Impl::CentroidRouteReadResult
PersistentSearchEngine::Impl::read_storage_centroid_route_publications() {
  const size_t shard_count = index.shards.size();
  if (control_bootstrapper == nullptr || shard_count == 0) return {};
  if (storage_centroid_route_descriptors.size() != shard_count ||
      centroid_route_snapshots.size() != shard_count) {
    throw std::logic_error(
      "storage centroid route descriptors were not initialized");
  }

  using Header = format::StorageCentroidRoutePublicationHeader;
  std::vector<NavigationRead> probe_requests(shard_count);
  std::vector<i32> probe_statuses(shard_count, -EIO);
  std::vector<Header> headers(shard_count);

  const auto transient_result = [&]() {
    // Retain the previous complete CPU/GPU route transaction. A concurrent
    // writer can tear a single header probe, but that probe is never installed
    // and the maintenance loop retries it at the 4 ms base delay.
    engine.telemetry_.centroid_route_snapshot_skips.fetch_add(
      1, std::memory_order_relaxed);
    return CentroidRouteReadResult{
      .shards = {},
      .snapshots = {},
      .transient = true,
    };
  };

  for (size_t shard = 0; shard < shard_count; ++shard) {
      const auto& descriptor = storage_centroid_route_descriptors[shard];
      if (descriptor.publication_bytes > storage_route_snapshot_stride ||
          descriptor.publication_bytes > std::numeric_limits<u32>::max()) {
        throw std::logic_error(
          "storage centroid route publication exceeds RDMA snapshot capacity");
      }
      // Pack probe headers contiguously at the start of the route scratch.
      // Body reads happen only after every probe has been copied to host.
      probe_requests[shard] = NavigationRead{
        .remote_offset = descriptor.remote_offset,
        .destination_address = reinterpret_cast<u64>(
          d_storage_route_snapshots + shard * sizeof(Header)),
        .bytes = sizeof(Header),
        .memory_node = static_cast<u16>(shard),
      };
      probe_statuses[shard] = -EIO;
  }
  control_bootstrapper->read(probe_requests, probe_statuses);
  engine.telemetry_.centroid_route_probe_reads.fetch_add(
    shard_count, std::memory_order_relaxed);

  check_cuda(cudaMemcpy(
               headers.data(), d_storage_route_snapshots,
               headers.size() * sizeof(Header), cudaMemcpyDeviceToHost),
             "cudaMemcpy(storage centroid route probe headers)");

  std::vector<u32> changed_shards;
  changed_shards.reserve(shard_count);
  bool transient = false;
  for (size_t shard = 0; shard < shard_count; ++shard) {
      if (probe_statuses[shard] <= 0) {
        throw std::runtime_error(
          "storage centroid route probe RDMA read failed for shard " +
          std::to_string(shard));
      }
      const Header& header = headers[shard];
      if (header.sequence == 0 || (header.sequence & 1u) != 0) {
        transient = true;
        continue;
      }
      const CentroidRouteSnapshot& cached = centroid_route_snapshots[shard];
      std::string error;
      if (!validate_centroid_route_probe(
            header, storage_centroid_route_descriptors[shard],
            static_cast<u32>(shard), error)) {
        // Mutable header fields can straddle a concurrent publication even
        // when the first cache line still contains the previous even sequence.
        // Once a valid cache exists, retain it and retry rather than turning a
        // low-frequency control-plane race into a query-engine failure.
        if (cached.version != 0) {
          transient = true;
          continue;
        }
        throw std::runtime_error(
          "shard " + std::to_string(shard) + ": " + error);
      }
      const centroid_route_poll::PublicationIdentity cached_identity{
        .sequence = cached.publication_sequence,
        .version = cached.version,
        .body_checksum = cached.body_checksum,
        .vector_count = cached.vector_count,
        .live_entry_count = cached.live_entry_count,
      };
      const centroid_route_poll::PublicationIdentity observed_identity{
        .sequence = header.sequence,
        .version = header.shard_version,
        .body_checksum = header.body_checksum,
        .vector_count = header.vector_count,
        .live_entry_count = header.live_entry_count,
      };
      const auto action = centroid_route_poll::classify_probe(
        cached.version != 0, cached_identity, observed_identity);
      if (action == centroid_route_poll::ProbeAction::retry) {
        transient = true;
      } else if (action == centroid_route_poll::ProbeAction::read_body) {
        changed_shards.push_back(static_cast<u32>(shard));
      }
  }
  if (transient) return transient_result();
  if (changed_shards.empty()) return {};

  std::vector<NavigationRead> body_requests(changed_shards.size());
  std::vector<NavigationRead> body_sequence_requests(changed_shards.size());
  std::vector<i32> body_statuses(changed_shards.size(), -EIO);
  std::vector<i32> body_sequence_statuses(changed_shards.size(), -EIO);
  for (size_t update = 0; update < changed_shards.size(); ++update) {
      const u32 shard = changed_shards[update];
      const auto& descriptor = storage_centroid_route_descriptors[shard];
      body_requests[update] = NavigationRead{
        .remote_offset = descriptor.remote_offset,
        .destination_address = reinterpret_cast<u64>(
          d_storage_route_snapshots + update * storage_route_snapshot_stride),
        .bytes = static_cast<u32>(descriptor.publication_bytes),
        .memory_node = static_cast<u16>(shard),
      };
      body_sequence_requests[update] = NavigationRead{
        .remote_offset = descriptor.remote_offset +
          offsetof(Header, sequence),
        .destination_address = reinterpret_cast<u64>(
          d_storage_route_sequence_after + update),
        .bytes = sizeof(u64),
        .memory_node = static_cast<u16>(shard),
      };
  }
  control_bootstrapper->read(body_requests, body_statuses);
  control_bootstrapper->read(
    body_sequence_requests, body_sequence_statuses);
  engine.telemetry_.centroid_route_body_reads.fetch_add(
    changed_shards.size(), std::memory_order_relaxed);

  std::vector<byte_t> publication_bytes(
    changed_shards.size() * storage_route_snapshot_stride);
  std::vector<u64> body_sequences_after(changed_shards.size());
  check_cuda(cudaMemcpy(
               publication_bytes.data(), d_storage_route_snapshots,
               publication_bytes.size(), cudaMemcpyDeviceToHost),
             "cudaMemcpy(changed storage centroid route publications)");
  check_cuda(cudaMemcpy(
               body_sequences_after.data(), d_storage_route_sequence_after,
               body_sequences_after.size() * sizeof(u64),
               cudaMemcpyDeviceToHost),
             "cudaMemcpy(changed storage centroid route sequences)");

  CentroidRouteReadResult result;
  result.shards = changed_shards;
  result.snapshots.resize(changed_shards.size());
  for (size_t update = 0; update < changed_shards.size(); ++update) {
      const u32 shard = changed_shards[update];
      if (body_statuses[update] <= 0 ||
          body_sequence_statuses[update] <= 0) {
        throw std::runtime_error(
          "storage centroid route body RDMA read failed for shard " +
          std::to_string(shard));
      }
      const auto& descriptor = storage_centroid_route_descriptors[shard];
      const span<const byte_t> publication{
        publication_bytes.data() + update * storage_route_snapshot_stride,
        static_cast<size_t>(descriptor.publication_bytes)};
      const auto* header = reinterpret_cast<const Header*>(publication.data());
      const centroid_route_poll::PublicationIdentity probed_identity{
        .sequence = headers[shard].sequence,
        .version = headers[shard].shard_version,
        .body_checksum = headers[shard].body_checksum,
        .vector_count = headers[shard].vector_count,
        .live_entry_count = headers[shard].live_entry_count,
      };
      const centroid_route_poll::PublicationIdentity body_identity{
        .sequence = header->sequence,
        .version = header->shard_version,
        .body_checksum = header->body_checksum,
        .vector_count = header->vector_count,
        .live_entry_count = header->live_entry_count,
      };
      if (!centroid_route_poll::body_read_is_stable(
            probed_identity, body_identity, body_sequences_after[update])) {
        transient = true;
        break;
      }

      std::string error;
      if (!format::validate_storage_centroid_route_publication(
            publication, descriptor, shard, &error)) {
        if (error ==
              "storage centroid route snapshot overlaps publication" ||
            error == "storage centroid route publication checksum mismatch") {
          transient = true;
          break;
        }
        throw std::runtime_error(
          "shard " + std::to_string(shard) + ": " + error);
      }

      CentroidRouteSnapshot& snapshot = result.snapshots[update];
      snapshot.publication_sequence = header->sequence;
      snapshot.body_checksum = header->body_checksum;
      snapshot.version = header->shard_version;
      snapshot.vector_count = header->vector_count;
      snapshot.centroid.resize(config.dim);
      const void* centroid =
        format::storage_centroid_route_centroid_data(publication);
      const auto scalar_type = static_cast<format::CentroidScalarType>(
        header->centroid_scalar_type);
      if (scalar_type != format::CentroidScalarType::float32) {
        throw std::logic_error(
          "validated centroid route is not canonical float32");
      }
      std::copy_n(static_cast<const f32*>(centroid), config.dim,
                  snapshot.centroid.begin());
      const auto entries =
        format::storage_centroid_route_entries(publication);
      snapshot.live_entry_count = static_cast<u32>(entries.size());
      for (u32 entry = 0; entry < snapshot.live_entry_count; ++entry) {
        snapshot.entries[entry] = DeviceCentroidRouteEntry{
          .remote_node = entries[entry].remote_node,
          .generation = entries[entry].generation,
          .flags = entries[entry].flags,
        };
      }
  }
  return transient ? transient_result() : std::move(result);
}

PersistentSearchEngine::Impl::StorageRouteSyncResult
PersistentSearchEngine::Impl::synchronize_storage_routes() {
  CentroidRouteReadResult refresh =
    read_storage_centroid_route_publications();
  if (refresh.transient) return StorageRouteSyncResult::transient;
  if (refresh.shards.size() != refresh.snapshots.size()) {
    throw std::logic_error("centroid route refresh cardinality mismatch");
  }
  if (refresh.shards.empty()) {
    engine.telemetry_.centroid_route_unchanged_polls.fetch_add(
      1, std::memory_order_relaxed);
    return StorageRouteSyncResult::unchanged;
  }

  std::vector<u32> changed_shards;
  changed_shards.reserve(refresh.shards.size());
  for (size_t update = 0; update < refresh.shards.size(); ++update) {
    const u32 shard = refresh.shards[update];
    if (shard >= centroid_route_versions.size()) {
      throw std::logic_error("centroid route refresh shard is out of range");
    }
    const CentroidRouteSnapshot& snapshot = refresh.snapshots[update];
    const CentroidRouteSnapshot& current = centroid_route_snapshots[shard];
    if (snapshot.version < centroid_route_versions[shard]) {
      throw std::runtime_error(
        "storage centroid route version regressed for shard " +
        std::to_string(shard));
    }
    if (current.version != 0 &&
        snapshot.version == centroid_route_versions[shard] &&
        snapshot.body_checksum != current.body_checksum) {
      throw std::runtime_error(
        "storage centroid route contents changed without a version advance "
        "for shard " + std::to_string(shard));
    }
    if (current.version == 0 ||
        snapshot.version != centroid_route_versions[shard]) {
      changed_shards.push_back(shard);
    }
  }
  if (changed_shards.empty()) {
    // A writer may republish byte-identical contents with a new seqlock
    // sequence. Advance only the per-shard poll identities; do not copy or
    // rebuild the complete routing snapshot in that case.
    for (size_t update = 0; update < refresh.shards.size(); ++update) {
      centroid_route_snapshots[refresh.shards[update]] =
        std::move(refresh.snapshots[update]);
    }
    engine.telemetry_.centroid_route_unchanged_polls.fetch_add(
      1, std::memory_order_relaxed);
    return StorageRouteSyncResult::unchanged;
  }

  std::vector<CentroidRouteSnapshot> next_snapshots =
    centroid_route_snapshots;
  for (size_t update = 0; update < refresh.shards.size(); ++update) {
    next_snapshots[refresh.shards[update]] =
      std::move(refresh.snapshots[update]);
  }

  for (u32 shard = 0; shard < next_snapshots.size(); ++shard) {
    if (next_snapshots[shard].version == 0 ||
        next_snapshots[shard].centroid.size() != config.dim) {
      return StorageRouteSyncResult::transient;
    }
  }

  auto cpu_snapshot = std::make_shared<centroid_home::Snapshot>(
    next_snapshots.size());
  u64 live_entries = 0;
  for (u32 shard = 0; shard < next_snapshots.size(); ++shard) {
    (*cpu_snapshot)[shard] = centroid_home::ShardSnapshot{
      .vector_count = next_snapshots[shard].vector_count,
      .centroid = next_snapshots[shard].centroid,
      .live_entry_count = next_snapshots[shard].live_entry_count,
    };
    live_entries += next_snapshots[shard].live_entry_count;
  }
  if (live_entries == 0) {
    throw std::runtime_error(
      "storage centroid routes contain no query-routable live entry");
  }

  for (u32 update_index = 0; update_index < changed_shards.size();
       ++update_index) {
    const u32 shard = changed_shards[update_index];
    const CentroidRouteSnapshot& snapshot = next_snapshots[shard];
    centroid_route_updates_host[update_index] = CentroidRouteUpdate{
      .version = snapshot.version,
      .vector_count = snapshot.vector_count,
      .shard = shard,
      .live_entry_count = snapshot.live_entry_count,
      .entries = snapshot.entries,
    };
    std::copy(snapshot.centroid.begin(), snapshot.centroid.end(),
              centroid_route_centroid_updates_host +
                static_cast<size_t>(update_index) * config.dim);
  }

  submit_centroid_route_publication(CentroidRoutePublishDescriptor{
    .command_id = next_route_command_id.fetch_add(
      1, std::memory_order_relaxed),
    .update_count = static_cast<u32>(changed_shards.size()),
  });
  for (u32 shard : changed_shards) {
    centroid_route_versions[shard] = next_snapshots[shard].version;
  }
  centroid_route_snapshots = std::move(next_snapshots);
  std::atomic_store_explicit(
    &centroid_home_snapshot,
    std::shared_ptr<const centroid_home::Snapshot>(std::move(cpu_snapshot)),
    std::memory_order_release);
  engine.telemetry_.centroid_route_live_entries.store(
    live_entries, std::memory_order_relaxed);
  engine.telemetry_.centroid_route_publications.fetch_add(
    1, std::memory_order_relaxed);
  engine.telemetry_.centroid_route_shard_updates.fetch_add(
    changed_shards.size(), std::memory_order_relaxed);
  return StorageRouteSyncResult::changed;
}

void PersistentSearchEngine::Impl::initialize_storage_route_descriptors() {
  const std::vector<format::StorageControlBlock> controls =
    read_storage_controls();
  storage_centroid_route_descriptors.resize(controls.size());
  for (size_t shard = 0; shard < controls.size(); ++shard) {
    storage_centroid_route_descriptors[shard] = controls[shard].centroid_route;
  }
}

void PersistentSearchEngine::Impl::maintenance_loop() {
  bind_cuda_device("cudaSetDevice(GPU centroid-route maintenance)");
  centroid_route_poll::AdaptiveIdleBackoff idle_backoff{route_poll_salt};
  engine.telemetry_.centroid_route_poll_delay_us.store(
    std::chrono::duration_cast<std::chrono::microseconds>(
      idle_backoff.delay()).count(),
    std::memory_order_relaxed);
  while (!maintenance_shutdown.load(std::memory_order_acquire)) {
    {
      std::unique_lock<std::mutex> lock(maintenance_mutex);
      maintenance_cv.wait_for(lock, idle_backoff.delay(), [&] {
        return maintenance_shutdown.load(std::memory_order_acquire);
      });
    }
    if (maintenance_shutdown.load(std::memory_order_acquire)) break;
    try {
      const StorageRouteSyncResult route_result =
        synchronize_storage_routes();
      // A torn publication is retried promptly. Stable idle periods back off
      // to 64 ms; a change or torn read resets the base delay to 4 ms. The
      // per-client deterministic jitter prevents synchronized compute polls.
      idle_backoff.observe(route_result != StorageRouteSyncResult::unchanged);
      engine.telemetry_.centroid_route_poll_delay_us.store(
        std::chrono::duration_cast<std::chrono::microseconds>(
          idle_backoff.delay()).count(),
        std::memory_order_relaxed);
    } catch (const std::exception& error) {
      mark_unhealthy(
        std::string{"storage centroid-route maintenance failed: "} +
        error.what());
      break;
    }
  }
}

}  // namespace gpu_search
