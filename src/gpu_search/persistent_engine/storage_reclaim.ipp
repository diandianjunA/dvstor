  void validate_storage_control(const format::StorageControlBlock& control,
                                size_t shard) const {
    if (control.magic != format::kStorageControlMagic ||
        control.version != format::kStorageControlVersion ||
        control.header_bytes != sizeof(format::StorageControlBlock) ||
        control.shard_id != shard ||
        control.compute_client_count != compute_client_count ||
        control.dynamic_record_bytes != index.shards[shard].dynamic_record_bytes ||
        control.dynamic_hot_offset != index.shards[shard].dynamic_hot_offset ||
        control.dynamic_code_offset != index.shards[shard].dynamic_code_offset ||
        control.code_bytes != index.layout.code_bytes) {
      std::ostringstream message;
      message << "storage maintenance control mismatch for shard " << shard
              << ": expected{magic=0x" << std::hex
              << format::kStorageControlMagic << std::dec
              << ",version=" << format::kStorageControlVersion
              << ",header=" << sizeof(format::StorageControlBlock)
              << ",shard=" << shard
              << ",clients=" << compute_client_count
              << ",record=" << index.shards[shard].dynamic_record_bytes
              << ",hot=" << index.shards[shard].dynamic_hot_offset
              << ",dynamic_code=" << index.shards[shard].dynamic_code_offset
              << ",code=" << index.layout.code_bytes
              << "} actual{magic=0x" << std::hex << control.magic << std::dec
              << ",version=" << control.version
              << ",header=" << control.header_bytes
              << ",shard=" << control.shard_id
              << ",clients=" << control.compute_client_count
              << ",record=" << control.dynamic_record_bytes
              << ",hot=" << control.dynamic_hot_offset
              << ",dynamic_code=" << control.dynamic_code_offset
              << ",code=" << control.code_bytes
              << "}. Rebuild and restart every storage node from the current "
                 "dev branch before starting the compute node.";
      throw std::runtime_error(message.str());
    }
  }

  std::vector<format::StorageControlBlock> read_storage_controls() {
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

  void write_storage_reclaim_acks(std::span<const u64> sequences) {
    if (sequences.size() != index.shards.size()) {
      throw std::invalid_argument("storage reclaim ACK cardinality mismatch");
    }
    std::vector<NavigationWrite> requests(index.shards.size());
    std::vector<i32> statuses(index.shards.size(), -EIO);
    for (size_t shard = 0; shard < index.shards.size(); ++shard) {
      u64* device_ack =
        &d_control_snapshots[shard].reclaim_ack_sequences[compute_client_id];
      check_cuda(cudaMemcpy(device_ack, &sequences[shard], sizeof(u64),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(storage reclaim ACK)");
      requests[shard] = NavigationWrite{
        .remote_offset = index.shards[shard].control_remote_offset +
          offsetof(format::StorageControlBlock, reclaim_ack_sequences) +
          static_cast<u64>(compute_client_id) * sizeof(u64),
        .source_address = reinterpret_cast<u64>(device_ack),
        .bytes = sizeof(u64),
        .memory_node = static_cast<u16>(shard),
      };
    }
    control_bootstrapper->write(requests, statuses);
    for (size_t shard = 0; shard < statuses.size(); ++shard) {
      if (statuses[shard] <= 0) {
        throw std::runtime_error(
          "storage reclaim ACK write failed for shard " +
          std::to_string(shard));
      }
    }
  }

  void initialize_storage_reclaim_ack() {
    (void)read_storage_controls();
    pending_storage_reclaim_acks.resize(index.shards.size());
    enqueued_reclaim_ack_sequences.assign(index.shards.size(), 0);
    published_reclaim_ack_sequences.assign(index.shards.size(), 0);
    const std::vector<u64> reset_sequences(index.shards.size(), 0);
    write_storage_reclaim_acks(reset_sequences);
    std::cerr << "[gpu-search] storage reclaim RCU client=" << compute_client_id
              << '/' << compute_client_count << " ACK reset complete\n";
  }

  void enqueue_storage_reclaim_barriers() {
    std::lock_guard<std::mutex> snapshot_lock(query_snapshot_mutex);
    const u64 barrier = next_query_ticket.load(std::memory_order_acquire) - 1;
    for (size_t shard = 0; shard < safe_durable_sequences.size(); ++shard) {
      const u64 sequence = safe_durable_sequences[shard];
      if (sequence <= enqueued_reclaim_ack_sequences[shard]) continue;
      auto& queue = pending_storage_reclaim_acks[shard];
      if (!queue.empty() && queue.back().query_ticket_barrier == barrier) {
        queue.back().maintenance_sequence = sequence;
      } else {
        queue.push_back(PendingStorageReclaimAck{
          .maintenance_sequence = sequence,
          .query_ticket_barrier = barrier,
        });
      }
      enqueued_reclaim_ack_sequences[shard] = sequence;
    }
  }

  void publish_ready_storage_reclaim_acks() {
    if (!healthy.load(std::memory_order_acquire)) return;
    if (!retired_resident_pq_batches.empty()) return;
    std::vector<u64> targets = published_reclaim_ack_sequences;
    bool advanced = false;
    for (size_t shard = 0; shard < pending_storage_reclaim_acks.size(); ++shard) {
      auto& queue = pending_storage_reclaim_acks[shard];
      while (!queue.empty() &&
             query_ticket_barrier_passed(queue.front().query_ticket_barrier)) {
        targets[shard] = queue.front().maintenance_sequence;
        queue.pop_front();
        advanced = true;
      }
    }
    if (!advanced) return;
    write_storage_reclaim_acks(targets);
    published_reclaim_ack_sequences = std::move(targets);
    engine.telemetry_.storage_reclaim_ack_writes.fetch_add(
      1, std::memory_order_relaxed);
    engine.telemetry_.storage_reclaim_ack_sequence.store(
      *std::min_element(published_reclaim_ack_sequences.begin(),
                        published_reclaim_ack_sequences.end()),
      std::memory_order_relaxed);
  }

  std::vector<DeltaMutation> retire_durable_delta() {
    if (control_bootstrapper == nullptr || index.shards.empty()) return {};
    const std::vector<format::StorageControlBlock> controls =
      read_storage_controls();
    if (durable_sequence_history.size() != index.shards.size()) {
      durable_sequence_history.resize(index.shards.size());
      observed_durable_sequences.assign(index.shards.size(), 0);
      safe_durable_sequences.assign(index.shards.size(), 0);
    }
    const auto now = std::chrono::steady_clock::now();
    const auto visibility_grace =
      std::chrono::microseconds(config.update_visibility_us);
    for (size_t shard = 0; shard < controls.size(); ++shard) {
      const auto& control = controls[shard];
      if (control.durable_maintenance_sequence > observed_durable_sequences[shard]) {
        observed_durable_sequences[shard] = control.durable_maintenance_sequence;
        durable_sequence_history[shard].emplace_back(
          control.durable_maintenance_sequence, now);
      }
      auto& history = durable_sequence_history[shard];
      while (!history.empty() && now - history.front().second >= visibility_grace) {
        safe_durable_sequences[shard] = history.front().first;
        history.pop_front();
      }
    }
    enqueue_storage_reclaim_barriers();
    return engine.delta_.retire_durable(
      safe_durable_sequences, delta_command_capacity);
  }

  void mark_durable_delta_records_locked(
      std::span<const DurableRetirement> retired) {
    std::vector<DeltaDurableUpdate> updates;
    std::vector<u32> retiring_slots;
    std::vector<ResidentPqEraseUpdate> retiring_resident_pq;
    std::unordered_set<u64> retained_resident_pq;
    retained_resident_pq.reserve(retired.size());
    for (const DurableRetirement& mutation : retired) {
      if (mutation.kind != service::storage_owner::MutationKind::erase &&
          mutation.remote_node != 0) {
        retained_resident_pq.insert(mutation.remote_node);
      }
      if (mutation.old_remote_node != 0 &&
          mutation.old_remote_node != mutation.remote_node) {
        const auto resident = resident_pq_slots_by_remote.find(
          mutation.old_remote_node);
        if (resident != resident_pq_slots_by_remote.end()) {
          retiring_resident_pq.push_back(ResidentPqEraseUpdate{
            .remote_node = mutation.old_remote_node,
            .slot = resident->second,
          });
        }
      }
      std::vector<u32> retained_superseded;
      const auto superseded = superseded_delta_slots.find(mutation.id);
      if (superseded != superseded_delta_slots.end()) {
        retained_superseded.reserve(superseded->second.size());
        for (u32 slot : superseded->second) {
          if (slot < delta_records_host.size() &&
              delta_records_host[slot].epoch <= mutation.epoch) {
            retiring_slots.push_back(slot);
          } else {
            retained_superseded.push_back(slot);
          }
        }
        if (retained_superseded.empty()) {
          superseded_delta_slots.erase(superseded);
        } else {
          superseded->second = std::move(retained_superseded);
        }
      }
      const auto latest = latest_delta_slot.find(mutation.id);
      if (latest != latest_delta_slot.end() &&
          latest->second < delta_records_host.size() &&
          delta_records_host[latest->second].epoch <= mutation.epoch) {
        DeviceDeltaRecord& record = delta_records_host[latest->second];
        if ((record.flags & (kDeltaDeleted | kDeltaDurable)) == 0 &&
            record.superseded_epoch == 0) {
          if (mutable_delta_entries == 0) {
            throw std::runtime_error("GPU mutable delta accounting underflow");
          }
          --mutable_delta_entries;
        }
        retiring_slots.push_back(latest->second);
        latest_delta_slot.erase(latest);
      }
    }
    std::sort(retiring_slots.begin(), retiring_slots.end());
    retiring_slots.erase(
      std::unique(retiring_slots.begin(), retiring_slots.end()),
      retiring_slots.end());
    for (u32 slot : retiring_slots) {
      const u64 remote_node = delta_records_host[slot].remote_node;
      if (remote_node == 0 || retained_resident_pq.contains(remote_node)) continue;
      const auto resident = resident_pq_slots_by_remote.find(remote_node);
      if (resident != resident_pq_slots_by_remote.end()) {
        retiring_resident_pq.push_back(ResidentPqEraseUpdate{
          .remote_node = remote_node,
          .slot = resident->second,
        });
      }
    }
    std::sort(retiring_resident_pq.begin(), retiring_resident_pq.end(),
              [](const ResidentPqEraseUpdate& lhs,
                 const ResidentPqEraseUpdate& rhs) {
                if (lhs.remote_node != rhs.remote_node) {
                  return lhs.remote_node < rhs.remote_node;
                }
                return lhs.slot < rhs.slot;
              });
    retiring_resident_pq.erase(
      std::unique(retiring_resident_pq.begin(), retiring_resident_pq.end(),
                  [](const ResidentPqEraseUpdate& lhs,
                     const ResidentPqEraseUpdate& rhs) {
                    return lhs.remote_node == rhs.remote_node &&
                      lhs.slot == rhs.slot;
                  }),
      retiring_resident_pq.end());
    updates.reserve(retiring_slots.size());
    for (u32 slot : retiring_slots) {
      DeviceDeltaRecord& record = delta_records_host[slot];
      updates.push_back(DeltaDurableUpdate{
        .slot = slot,
        .epoch = record.epoch,
      });
    }
    for (size_t begin = 0; begin < updates.size(); begin += delta_command_capacity) {
      const size_t count = std::min<size_t>(
        delta_command_capacity, updates.size() - begin);
      std::memcpy(delta_durable_updates_host, updates.data() + begin,
                  count * sizeof(DeltaDurableUpdate));
      const u32 live_count = static_cast<u32>(delta_records_host.size());
      submit_delta_publication(DeltaPublishDescriptor{
        .command_id = next_delta_command_id.fetch_add(1, std::memory_order_relaxed),
        .final_count = live_count,
        .durable_count = static_cast<u32>(count),
      });
    }
    if (!retiring_slots.empty() || !retiring_resident_pq.empty()) {
      for (u32 slot : retiring_slots) {
        DeviceDeltaRecord& record = delta_records_host[slot];
        record.flags |= kDeltaDurable;
        if (record.superseded_epoch == 0) record.superseded_epoch = record.epoch;
        if (record.base_ordinal != kBaseOverrideEmpty) {
          const auto override = base_override_epochs.find(record.base_ordinal);
          if (override != base_override_epochs.end() &&
              override->second <= record.epoch) {
            base_override_epochs.erase(override);
          }
        }
      }
      std::lock_guard<std::mutex> snapshot_lock(query_snapshot_mutex);
      const u64 barrier = next_query_ticket.load(std::memory_order_acquire) - 1;
      retired_delta_batches.push_back(RetiredDeltaBatch{
        .query_ticket_barrier = barrier,
        .slots = std::move(retiring_slots),
      });
      if (!retiring_resident_pq.empty()) {
        retired_resident_pq_batches.push_back(RetiredResidentPqBatch{
          .query_ticket_barrier = barrier,
          .entries = std::move(retiring_resident_pq),
        });
      }
      reclaim_retired_delta_slots_locked();
    }
    engine.telemetry_.delta_mutable_entries.store(
      mutable_delta_entries, std::memory_order_relaxed);
    engine.telemetry_.delta_durable_entries.store(
      durable_delta_entries, std::memory_order_relaxed);
    engine.telemetry_.delta_entries_retired.fetch_add(
      updates.size(), std::memory_order_relaxed);
  }

  void maintenance_loop() {
    bind_cuda_device("cudaSetDevice(GPU navigation maintenance)");
    const auto period = std::chrono::milliseconds(std::max<u32>(
      1, std::min<u32>(config.gpu_delta_maintenance_period_ms,
                       std::max<u32>(1, config.update_visibility_us / 1000))));
    while (!maintenance_shutdown.load(std::memory_order_acquire)) {
      {
        std::unique_lock<std::mutex> lock(maintenance_mutex);
        maintenance_cv.wait_for(lock, period, [&] { return maintenance_shutdown.load(); });
      }
      if (maintenance_shutdown.load()) break;
      std::vector<DeltaMutation> retired;
      try {
        std::lock_guard<std::mutex> publish_lock(engine.mutation_publish_mutex_);
        retired = retire_durable_delta();
        for (const DeltaMutation& mutation : retired) {
          pending_durable_retirements.emplace(
            mutation.epoch,
            DurableRetirement{
              .id = mutation.id,
              .kind = mutation.kind,
              .epoch = mutation.epoch,
              .remote_node = mutation.remote_node,
              .old_remote_node = mutation.old_remote_node,
            });
        }
        std::vector<DurableRetirement> snapshot_safe;
        snapshot_safe.reserve(std::min<size_t>(
          delta_command_capacity, pending_durable_retirements.size()));
        {
          std::lock_guard<std::mutex> snapshot_lock(query_snapshot_mutex);
          while (!pending_durable_retirements.empty() &&
                 snapshot_safe.size() < delta_command_capacity) {
            auto oldest = pending_durable_retirements.begin();
            if (!durable_snapshot_safe(oldest->first)) break;
            snapshot_safe.push_back(std::move(oldest->second));
            pending_durable_retirements.erase(oldest);
          }
        }
        {
          std::lock_guard<std::mutex> delta_lock(delta_mutex);
          if (!snapshot_safe.empty()) {
            mark_durable_delta_records_locked(snapshot_safe);
          }
          reclaim_retired_delta_slots_locked();
        }
        publish_ready_storage_reclaim_acks();
      } catch (const std::exception& error) {
        mark_unhealthy(std::string{"storage maintenance watermark failed: "} + error.what());
        break;
      }
    }
  }

