#include "gpu_search/persistent_engine/impl.hh"
#include "gpu_search/persistent_engine/cuda_helpers.hh"

namespace gpu_search {

using namespace persistent_engine_detail;
void PersistentSearchEngine::Impl::submit_delta_publication(const DeltaPublishDescriptor& descriptor) {
  const auto timeout = std::chrono::milliseconds(std::clamp<u32>(
    config.storage_owner_rpc_timeout_ms, 1000, 5000));
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!delta_submissions.try_push(descriptor)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error("persistent GPU delta command queue is not making progress");
    }
    std::this_thread::yield();
  }

  DeltaPublishCompletion completion{};
  while (!delta_completions.try_pop(completion)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error("persistent GPU delta publication timed out");
    }
    std::this_thread::yield();
  }
  if (completion.command_id != descriptor.command_id || completion.status != 0 ||
      completion.final_count != descriptor.final_count) {
    throw std::runtime_error(
      "persistent GPU delta publication failed: command=" +
      std::to_string(completion.command_id) + " status=" +
      std::to_string(completion.status) + " count=" +
      std::to_string(completion.final_count));
  }
}

size_t PersistentSearchEngine::Impl::active_resident_pq_slots_locked() const {
  return resident_pq_slots_by_remote.size();
}

u32 PersistentSearchEngine::Impl::allocate_resident_pq_slot_locked(u64 remote_node) {
  if (remote_node == 0) {
    throw std::runtime_error(
      "cannot allocate resident GPU PQ for a null remote node");
  }
  if (resident_pq_slots_by_remote.contains(remote_node)) {
    throw std::runtime_error(
      "storage reused a dynamic remote node before its resident GPU PQ was reclaimed");
  }
  u32 slot = UINT32_MAX;
  if (!free_resident_pq_slots.empty()) {
    slot = free_resident_pq_slots.back();
    free_resident_pq_slots.pop_back();
  } else if (resident_pq_high_watermark < resident_pq_capacity) {
    slot = resident_pq_high_watermark++;
  } else {
    throw MutationCapacityError(
      "resident GPU PQ tier is full; increase --gpu-resident-pq-budget-mb "
      "or consolidate dynamic vectors into a new base generation");
  }
  resident_pq_slots_by_remote.emplace(remote_node, slot);
  const u64 live = active_resident_pq_slots_locked();
  engine.telemetry_.resident_pq_entries.store(live, std::memory_order_relaxed);
  u64 peak = engine.telemetry_.resident_pq_peak_entries.load(
    std::memory_order_relaxed);
  while (peak < live &&
         !engine.telemetry_.resident_pq_peak_entries.compare_exchange_weak(
           peak, live, std::memory_order_relaxed)) {}
  return slot;
}

void PersistentSearchEngine::Impl::upload_records_locked(std::span<DeltaMutation> mutations,
                           std::span<const u64> invalidation_keys) {
  const auto prepare_started = std::chrono::steady_clock::now();
  bind_cuda_device("cudaSetDevice(GPU navigation delta publication)");
  (void)cudaGetLastError();
  const size_t available_slots = free_delta_slots.size() +
    (delta_capacity - delta_records_host.size());
  if (mutations.size() > available_slots) {
    throw std::runtime_error("GPU navigation delta live set exceeds its configured capacity");
  }
  const size_t vector_bytes = VamanaNode::vector_bytes();
  std::vector<DeviceDeltaRecord> records;
  std::vector<u32> destination_slots;
  std::vector<byte_t> vectors(static_cast<size_t>(mutations.size()) * vector_bytes);
  records.reserve(mutations.size());
  destination_slots.reserve(mutations.size());
  std::unordered_map<u32, size_t> staged_record_indices;
  std::vector<DeltaSupersedeUpdate> superseded_updates;
  std::vector<DeltaOverrideUpdate> override_updates;
  std::vector<f32> decoded(config.dim);
  for (size_t mutation_index = 0; mutation_index < mutations.size(); ++mutation_index) {
    DeltaMutation& mutation = mutations[mutation_index];
    bool decoded_ready = false;
    u32 slot = UINT32_MAX;
    if (!free_delta_slots.empty()) {
      slot = free_delta_slots.back();
      free_delta_slots.pop_back();
    } else {
      slot = static_cast<u32>(delta_records_host.size());
      delta_records_host.emplace_back();
    }
    const auto previous = latest_delta_slot.find(mutation.id);
    if (previous != latest_delta_slot.end()) {
      DeviceDeltaRecord& previous_record = delta_records_host[previous->second];
      if (previous_record.superseded_epoch == 0 &&
          (previous_record.flags & kDeltaDeleted) == 0) {
        if ((previous_record.flags & kDeltaDurable) != 0) {
          --durable_delta_entries;
        } else {
          --mutable_delta_entries;
        }
      }
      previous_record.superseded_epoch = mutation.epoch;
      superseded_delta_slots[mutation.id].push_back(previous->second);
      const auto staged = staged_record_indices.find(previous->second);
      if (staged != staged_record_indices.end()) {
        records[staged->second].superseded_epoch = mutation.epoch;
      } else {
        superseded_updates.push_back(DeltaSupersedeUpdate{
          .slot = previous->second,
          .epoch = mutation.epoch,
        });
      }
    }
    const bool deleted = mutation.kind == service::storage_owner::MutationKind::erase;
    const u64 record_remote = mutation.remote_node != 0
      ? mutation.remote_node : mutation.old_remote_node;
    const u32 route_shard = static_cast<u32>(record_remote >> 48);
    if (record_remote == 0 || route_shard >= index.shards.size() ||
        route_shard != mutation.owner_storage) {
      throw std::runtime_error(
        "storage returned an invalid physical owner for GPU dynamic routing");
    }
    // Reuse the graph-address validator so an acknowledged but misaligned
    // dynamic pointer can never enter either the delta map or route overlay.
    (void)graph_cache_key(record_remote);
    u32 bucket = 0;
    if (!deleted) {
      const auto hinted = anchor_buckets_by_raw.find(mutation.anchor_hint);
      if (hinted == anchor_buckets_by_raw.end()) {
        if (!decoded_ready) {
          decode_mutation_payload(mutation, decoded);
          decoded_ready = true;
        }
        bucket = nearest_anchor(decoded, record_remote);
      } else {
        bucket = hinted->second;
      }
    }
    u32 base_ordinal = kBaseOverrideEmpty;
    if (format::remote_to_ordinal(
          index, RemotePtr{mutation.old_remote_node}, base_ordinal)) {
      const auto [it, inserted] =
        base_override_epochs.emplace(base_ordinal, mutation.epoch);
      if (inserted) {
        override_updates.push_back(DeltaOverrideUpdate{
          .ordinal = base_ordinal,
          .epoch = mutation.epoch,
        });
      } else if (mutation.epoch < it->second) {
        it->second = mutation.epoch;
        override_updates.push_back(DeltaOverrideUpdate{
          .ordinal = base_ordinal,
          .epoch = mutation.epoch,
        });
      }
    } else {
      base_ordinal = kBaseOverrideEmpty;
    }
    DeviceDeltaRecord record{
      .id = mutation.id,
      .generation = std::max<u32>(1, mutation.generation),
      .flags = (deleted ? kDeltaDeleted : 0u) |
        (mutation.durable ? kDeltaDurable : 0u),
      .base_ordinal = base_ordinal,
      .epoch = mutation.epoch,
      .remote_node = record_remote,
      .anchor_bucket = bucket,
      .resident_pq_slot = deleted
        ? UINT32_MAX : allocate_resident_pq_slot_locked(record_remote),
    };
    delta_records_host[slot] = record;
    records.push_back(record);
    destination_slots.push_back(slot);
    staged_record_indices.emplace(slot, records.size() - 1);
    latest_delta_slot[mutation.id] = slot;
    if (!deleted) {
      if (mutation.durable) {
        ++durable_delta_entries;
      } else {
        ++mutable_delta_entries;
      }
    }
    byte_t* stored_vector = vectors.data() + mutation_index * vector_bytes;
    if (deleted) {
      std::memset(stored_vector, 0, vector_bytes);
    } else if (mutation.vector.size() == vector_bytes) {
      std::memcpy(stored_vector, mutation.vector.data(), vector_bytes);
    } else {
      if (!decoded_ready) {
        decode_mutation_payload(mutation, decoded);
        decoded_ready = true;
      }
      encode_float_vector_to_storage(decoded.data(), config.dim,
                                     config.resolved_vector_dtype(), stored_vector);
    }
  }

  if (records.size() > delta_command_capacity ||
      superseded_updates.size() > delta_command_capacity ||
      override_updates.size() > delta_command_capacity ||
      invalidation_keys.size() > graph_invalidation_capacity) {
    throw std::runtime_error("GPU navigation delta control batch exceeds capacity");
  }

  std::memcpy(delta_staging_records_host, records.data(),
              records.size() * sizeof(DeviceDeltaRecord));
  std::memcpy(delta_staging_slots_host, destination_slots.data(),
              destination_slots.size() * sizeof(u32));
  std::memcpy(delta_staging_vectors_host, vectors.data(), vectors.size());
  if (!superseded_updates.empty()) {
    std::memcpy(delta_supersede_updates_host, superseded_updates.data(),
                superseded_updates.size() * sizeof(DeltaSupersedeUpdate));
  }
  if (!override_updates.empty()) {
    std::memcpy(delta_override_updates_host, override_updates.data(),
                override_updates.size() * sizeof(DeltaOverrideUpdate));
  }
  if (!invalidation_keys.empty()) {
    std::memcpy(graph_invalidation_keys_host, invalidation_keys.data(),
                invalidation_keys.size() * sizeof(u64));
  }
  const u32 count = static_cast<u32>(delta_records_host.size());
  const auto command_started = std::chrono::steady_clock::now();
  engine.telemetry_.publication_prepare_ns_total.fetch_add(
    static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      command_started - prepare_started).count()), std::memory_order_relaxed);
  submit_delta_publication(DeltaPublishDescriptor{
    .command_id = next_delta_command_id.fetch_add(1, std::memory_order_relaxed),
    .record_count = static_cast<u32>(records.size()),
    .final_count = count,
    .invalidation_count = static_cast<u32>(invalidation_keys.size()),
    .superseded_count = static_cast<u32>(superseded_updates.size()),
    .override_count = static_cast<u32>(override_updates.size()),
  });
  refresh_anchor_graph_records(invalidation_keys);
  engine.telemetry_.publication_command_ns_total.fetch_add(
    static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - command_started).count()),
    std::memory_order_relaxed);
  engine.telemetry_.delta_physical_entries.store(
    count - free_delta_slots.size(), std::memory_order_relaxed);
  engine.telemetry_.delta_mutable_entries.store(
    mutable_delta_entries, std::memory_order_relaxed);
  engine.telemetry_.delta_durable_entries.store(
    durable_delta_entries, std::memory_order_relaxed);
}

size_t PersistentSearchEngine::Impl::upload_mutations(std::span<DeltaMutation> mutations, u64 epoch,
                        std::span<const u64> invalidated_graph_nodes) {
  if (mutations.empty()) return 0;
  const std::vector<u64> invalidation_keys = graph_cache_keys(invalidated_graph_nodes);
  std::lock_guard<std::mutex> lock(delta_mutex);
  reclaim_retired_delta_slots_locked();
  const size_t active_slots = active_delta_slots_locked();
  const size_t hard_watermark = static_cast<size_t>(delta_capacity) * 9 / 10;
  if (active_slots + mutations.size() > hard_watermark) {
    throw MutationCapacityError(
      "bounded GPU update tier reached its hard watermark; "
      "storage maintenance has not retired updates quickly enough");
  }
  for (DeltaMutation& mutation : mutations) {
    mutation.epoch = epoch;
  }
  upload_records_locked(mutations, invalidation_keys);
  return invalidation_keys.size();
}

size_t PersistentSearchEngine::Impl::active_delta_slots_locked() const {
  return delta_records_host.size() - free_delta_slots.size();
}

bool PersistentSearchEngine::Impl::query_ticket_barrier_passed(u64 barrier) const {
  for (u32 slot = 0; slot < query_slots; ++slot) {
    const u64 ticket = active_query_tickets[slot].load(std::memory_order_acquire);
    if (ticket != 0 && ticket <= barrier) return false;
  }
  return true;
}

bool PersistentSearchEngine::Impl::durable_snapshot_safe(u64 durable_epoch) const {
  for (u32 slot = 0; slot < query_slots; ++slot) {
    const u64 encoded_snapshot =
      active_query_snapshots[slot].load(std::memory_order_acquire);
    if (encoded_snapshot != 0 && encoded_snapshot - 1 < durable_epoch) {
      return false;
    }
  }
  return true;
}

void PersistentSearchEngine::Impl::reclaim_retired_delta_slots_locked() {
  u64 reclaimed = 0;
  while (!retired_delta_batches.empty() &&
         query_ticket_barrier_passed(
           retired_delta_batches.front().query_ticket_barrier)) {
    RetiredDeltaBatch batch = std::move(retired_delta_batches.front());
    retired_delta_batches.pop_front();
    reclaimed += batch.slots.size();
    free_delta_slots.insert(free_delta_slots.end(),
                            batch.slots.begin(), batch.slots.end());
  }
  u64 resident_pq_reclaimed = 0;
  while (!retired_resident_pq_batches.empty() &&
         query_ticket_barrier_passed(
           retired_resident_pq_batches.front().query_ticket_barrier)) {
    RetiredResidentPqBatch batch =
      std::move(retired_resident_pq_batches.front());
    retired_resident_pq_batches.pop_front();
    for (size_t begin = 0; begin < batch.entries.size();
         begin += delta_command_capacity) {
      const size_t count = std::min<size_t>(
        delta_command_capacity, batch.entries.size() - begin);
      std::memcpy(resident_pq_erase_updates_host,
                  batch.entries.data() + begin,
                  count * sizeof(ResidentPqEraseUpdate));
      submit_delta_publication(DeltaPublishDescriptor{
        .command_id = next_delta_command_id.fetch_add(
          1, std::memory_order_relaxed),
        .final_count = static_cast<u32>(delta_records_host.size()),
        .resident_pq_erase_count = static_cast<u32>(count),
      });
      for (size_t index = 0; index < count; ++index) {
        const ResidentPqEraseUpdate& update = batch.entries[begin + index];
        const auto resident = resident_pq_slots_by_remote.find(
          update.remote_node);
        if (resident == resident_pq_slots_by_remote.end() ||
            resident->second != update.slot) {
          continue;
        }
        resident_pq_slots_by_remote.erase(resident);
        free_resident_pq_slots.push_back(update.slot);
        ++resident_pq_reclaimed;
      }
    }
  }
  if (reclaimed != 0) {
    engine.telemetry_.delta_reclaim_batches.fetch_add(1, std::memory_order_relaxed);
  }
  if (resident_pq_reclaimed != 0) {
    engine.telemetry_.resident_pq_reclaimed.fetch_add(
      resident_pq_reclaimed, std::memory_order_relaxed);
  }
  engine.telemetry_.delta_physical_entries.store(
    active_delta_slots_locked(), std::memory_order_relaxed);
  engine.telemetry_.resident_pq_entries.store(
    active_resident_pq_slots_locked(), std::memory_order_relaxed);
}

}  // namespace gpu_search
