#include "gpu_search/persistent_engine/impl.hh"
#include "gpu_search/persistent_engine/cuda_helpers.hh"

namespace gpu_search {

using namespace persistent_engine_detail;
void PersistentSearchEngine::Impl::stream_codes_to_gpu(NavigationBootstrapper& source) {
  const u64 window_bytes = static_cast<u64>(config.gpu_bootstrap_window_mb) << 20;
  std::vector<NavigationRead> requests;
  std::vector<i32> statuses;
  requests.reserve(config.gpu_bootstrap_windows);
  u64 streamed = 0;
  for (const format::ShardRegion& shard : index.shards) {
    for (u64 offset = 0; offset < shard.code_bytes;) {
      requests.clear();
      for (u32 window = 0; window < config.gpu_bootstrap_windows &&
           offset < shard.code_bytes; ++window) {
        const u32 bytes = static_cast<u32>(std::min<u64>(
          window_bytes, shard.code_bytes - offset));
        requests.push_back(NavigationRead{
          .remote_offset = shard.code_remote_offset + offset,
          .destination_address = reinterpret_cast<u64>(d_pq_codes +
            shard.ordinal_base * code_bytes + offset),
          .bytes = bytes,
          .memory_node = static_cast<u16>(shard.memory_node),
        });
        offset += bytes;
      }
      statuses.assign(requests.size(), -EIO);
      source.read(requests, statuses);
      for (size_t request_index = 0; request_index < statuses.size(); ++request_index) {
        if (statuses[request_index] <= 0) {
          const NavigationRead& request = requests[request_index];
          throw std::runtime_error(
            "RDMA PQ code bootstrap failed: status=" +
            std::to_string(statuses[request_index]) + " shard=" +
            std::to_string(request.memory_node) + " remote_offset=" +
            std::to_string(request.remote_offset) + " bytes=" +
            std::to_string(request.bytes) + " destination=" +
            std::to_string(request.destination_address));
        }
      }
      for (const NavigationRead& request : requests) streamed += request.bytes;
    }
  }
  const u64 expected = index.layout.num_nodes * code_bytes;
  if (streamed != expected) throw std::runtime_error("GPU PQ code bootstrap size mismatch");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(GPU PQ bootstrap)");

  struct AuditSample {
    u32 shard{};
    u64 slot{};
    u64 ordinal{};
  };
  std::vector<AuditSample> samples;
  samples.reserve(index.shards.size() * 3);
  for (const format::ShardRegion& shard : index.shards) {
    const std::array<u64, 3> shard_slots{0, shard.node_count / 2, shard.node_count - 1};
    for (size_t sample_index = 0; sample_index < shard_slots.size(); ++sample_index) {
      if (sample_index != 0 && shard_slots[sample_index] == shard_slots[sample_index - 1]) {
        continue;
      }
      const u64 slot = shard_slots[sample_index];
      samples.push_back(AuditSample{
        .shard = shard.memory_node,
        .slot = slot,
        .ordinal = shard.ordinal_base + slot,
      });
    }
  }
  std::vector<byte_t> authoritative(code_bytes);
  std::vector<byte_t> resident(code_bytes);
  for (size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
    const AuditSample& sample = samples[sample_index];
    const format::ShardRegion& shard = index.shards[sample.shard];
    requests.assign(1, NavigationRead{
      .remote_offset = shard.code_remote_offset + sample.slot * code_bytes,
      .destination_address = reinterpret_cast<u64>(d_exact_records),
      .bytes = code_bytes,
      .memory_node = static_cast<u16>(sample.shard),
    });
    statuses.assign(1, -EIO);
    source.read(requests, statuses);
    if (statuses.front() <= 0) {
      throw std::runtime_error(
        "GPU PQ ordinal audit RDMA read failed: shard=" +
        std::to_string(sample.shard) + " slot=" +
        std::to_string(sample.slot) + " status=" +
        std::to_string(statuses.front()));
    }
    check_cuda(cudaMemcpy(authoritative.data(), d_exact_records, authoritative.size(),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(GPU PQ audit source)");
    check_cuda(cudaMemcpy(
      resident.data(),
      d_pq_codes + sample.ordinal * code_bytes,
      resident.size(), cudaMemcpyDeviceToHost),
      "cudaMemcpy(GPU PQ audit resident)");
    if (!std::equal(resident.begin(), resident.end(), authoritative.begin())) {
      throw std::runtime_error(
        "GPU PQ ordinal mapping mismatch: shard=" +
        std::to_string(sample.shard) + " slot=" +
        std::to_string(sample.slot) + " ordinal=" +
        std::to_string(sample.ordinal));
    }
  }
  std::cerr << "[gpu-search] streamed " << streamed
            << " PQ bytes directly into final GPU storage; ordinal audit passed for "
            << samples.size() << " entries\n";
}

void PersistentSearchEngine::Impl::stream_anchor_graph_to_gpu(NavigationBootstrapper& source) {
  if (anchor_graph_keys_host.empty()) {
    std::cerr << "[gpu-search] static fallback route graph disabled\n";
    return;
  }
  constexpr size_t kBootstrapBatch = 4096;
  std::vector<NavigationRead> requests;
  std::vector<i32> statuses;
  requests.reserve(kBootstrapBatch);
  for (size_t begin = 0; begin < anchor_graph_keys_host.size();
       begin += kBootstrapBatch) {
    const size_t end = std::min(begin + kBootstrapBatch,
                                anchor_graph_keys_host.size());
    requests.clear();
    for (size_t slot = begin; slot < end; ++slot) {
      const u64 key = anchor_graph_keys_host[slot];
      const u32 shard = static_cast<u32>(key >> 48);
      if (shard >= index.shards.size()) {
        throw std::runtime_error("anchor route graph key has an invalid shard");
      }
      requests.push_back(NavigationRead{
        .remote_offset = (key << 16) >> 16,
        .destination_address = reinterpret_cast<u64>(
          d_anchor_graph_records +
          slot * index.layout.graph_entry_bytes),
        .bytes = index.layout.graph_entry_bytes,
        .memory_node = static_cast<u16>(shard),
      });
    }
    statuses.assign(requests.size(), -EIO);
    source.read(requests, statuses);
    for (size_t request = 0; request < statuses.size(); ++request) {
      if (statuses[request] <= 0) {
        throw std::runtime_error(
          "anchor route graph bootstrap failed: slot=" +
          std::to_string(begin + request) + " status=" +
          std::to_string(statuses[request]));
      }
    }
  }

  const size_t audit_count = std::min<size_t>(15, anchor_graph_keys_host.size());
  std::vector<byte_t> record(index.layout.graph_entry_bytes);
  for (size_t audit = 0; audit < audit_count; ++audit) {
    const size_t slot = audit_count == 1 ? 0 :
      audit * (anchor_graph_keys_host.size() - 1) / (audit_count - 1);
    check_cuda(cudaMemcpy(
                 record.data(),
                 d_anchor_graph_records + slot * index.layout.graph_entry_bytes,
                 record.size(), cudaMemcpyDeviceToHost),
               "cudaMemcpy(anchor route graph audit)");
    const u16 expected = vamana::hot_graph::load_u16_le(record.data() + 2);
    const u16 actual = vamana::hot_graph::checksum16(record.data(), record.size());
    if (record[0] > index.layout.graph_degree || expected != actual) {
      throw std::runtime_error(
        "anchor route graph audit failed at slot " + std::to_string(slot));
    }
  }
  std::cerr << "[gpu-search] resident static fallback graph records="
            << anchor_graph_keys_host.size() << " bytes="
            << anchor_graph_keys_host.size() * index.layout.graph_entry_bytes
            << " audit=" << audit_count << '\n';
}

void PersistentSearchEngine::Impl::clear_delta_device_state(cudaStream_t stream) {
  bind_cuda_device("cudaSetDevice(GPU navigation delta reset)");
  check_cuda(cudaMemsetAsync(d_delta_records, 0,
                             static_cast<size_t>(delta_capacity) * sizeof(DeviceDeltaRecord),
                             stream),
             "cudaMemset(navigation delta records)");
  check_cuda(cudaMemsetAsync(d_delta_next, 0xff,
                             static_cast<size_t>(delta_capacity) * sizeof(u32), stream),
             "cudaMemset(navigation delta links)");
  check_cuda(cudaMemsetAsync(d_delta_prev, 0xff,
                             static_cast<size_t>(delta_capacity) * sizeof(u32), stream),
             "cudaMemset(navigation delta reverse links)");
  check_cuda(cudaMemsetAsync(d_delta_remote_positions, 0xff,
                             static_cast<size_t>(delta_capacity) * sizeof(u32), stream),
             "cudaMemset(navigation delta remote positions)");
  check_cuda(cudaMemsetAsync(d_base_override_keys, 0xff,
                             static_cast<size_t>(delta_table_capacity) * sizeof(u32), stream),
             "cudaMemset(navigation override keys)");
  check_cuda(cudaMemsetAsync(d_base_override_epochs, 0,
                             static_cast<size_t>(delta_table_capacity) * sizeof(u64), stream),
             "cudaMemset(navigation override epochs)");
  check_cuda(cudaMemsetAsync(d_permanent_override_bits, 0,
                             static_cast<size_t>(permanent_override_words) * sizeof(u32),
                             stream),
             "cudaMemset(navigation permanent override bits)");
  check_cuda(cudaMemsetAsync(d_delta_remote_keys, 0,
                             static_cast<size_t>(delta_table_capacity) * sizeof(u64), stream),
             "cudaMemset(navigation remote keys)");
  check_cuda(cudaMemsetAsync(d_delta_remote_slots, 0xff,
                             static_cast<size_t>(delta_table_capacity) * sizeof(u32), stream),
             "cudaMemset(navigation remote slots)");
  check_cuda(cudaMemsetAsync(d_delta_count, 0, sizeof(u32), stream),
             "cudaMemset(navigation delta count)");
  if (d_delta_bucket_heads != nullptr) {
    check_cuda(cudaMemsetAsync(d_delta_bucket_heads, 0xff,
                               static_cast<size_t>(anchor_table.count()) * sizeof(u32),
                               stream),
               "cudaMemset(navigation delta buckets)");
  }
  check_cuda(cudaStreamSynchronize(stream),
             "cudaStreamSynchronize(navigation delta reset)");
}

void PersistentSearchEngine::Impl::start_persistent_kernel() {
  bind_cuda_device("cudaSetDevice(GPU navigation kernel start)");
  *stop_host = 0;
  *direct_disabled_host = 0;
  *direct_error_host = 0;
  check_cuda(cudaMemset(stop_device, 0, sizeof(u32)),
             "cudaMemset(GPU navigation start flag)");
  check_cuda(cudaMemset(direct_disabled_device, 0, sizeof(u32)),
             "cudaMemset(GPU navigation direct failure flag)");
  check_cuda(cudaMemset(direct_error_device, 0, sizeof(i32)),
             "cudaMemset(GPU navigation direct error)");
  (void)cudaGetLastError();
  std::fill_n(direct_owner_phases_host, direct_batch_queue_count, 0u);
  *query_kernel_ready_host = 0;
  *dispatcher_kernel_ready_host = 0;
  *control_kernel_ready_host = 0;
  std::atomic_thread_fence(std::memory_order_release);
  PersistentKernelParams launch_params = kernel_params;
  launch_params.direct_owner_block_count = owner_kernel_blocks;
  launch_params.query_block_count = kernel_blocks;
  launch_params.query_kernel_ready_count = d_query_kernel_ready;
  launch_params.dispatcher_kernel_ready_count = d_dispatcher_kernel_ready;
  launch_params.control_kernel_ready_count = d_control_kernel_ready;
  const u32 total_blocks = owner_kernel_blocks + kernel_blocks + 2;
  launch_persistent_search(kernel_stream, launch_params, total_blocks,
                           kPersistentQueryThreads);
  check_cuda(cudaGetLastError(), "launch_persistent_search(unified navigation)");

  const auto ready_deadline = std::chrono::steady_clock::now() +
    std::chrono::seconds(3);
  u32 ready_owners = 0;
  for (;;) {
    ready_owners = 0;
    for (u32 qp = 0; qp < direct_batch_queue_count; ++qp) {
      ready_owners +=
        *reinterpret_cast<volatile u32*>(direct_owner_phases_host + qp) == 1
          ? 1u : 0u;
    }
    const u32 ready_queries =
      *reinterpret_cast<volatile u32*>(query_kernel_ready_host);
    const u32 ready_dispatchers =
      *reinterpret_cast<volatile u32*>(dispatcher_kernel_ready_host);
    const u32 ready_controls =
      *reinterpret_cast<volatile u32*>(control_kernel_ready_host);
    if (ready_owners == direct_batch_queue_count &&
        ready_queries == kernel_blocks && ready_dispatchers == 1 &&
        ready_controls == 1) {
      break;
    }
    if (std::chrono::steady_clock::now() >= ready_deadline) {
      u32 first_owner_phase = 0;
      for (u32 qp = 0; qp < direct_batch_queue_count; ++qp) {
        const u32 phase =
          *reinterpret_cast<volatile u32*>(direct_owner_phases_host + qp);
        if (phase != 1) {
          first_owner_phase = phase;
          break;
        }
      }
      *stop_host = 1;
      (void)cudaMemcpyAsync(stop_device, stop_host, sizeof(u32),
                            cudaMemcpyHostToDevice, rdma_stream);
      (void)cudaStreamSynchronize(rdma_stream);
      (void)cudaStreamSynchronize(kernel_stream);
      throw std::runtime_error(
        "unified GPU grid did not become fully resident: owners=" +
        std::to_string(ready_owners) + "/" +
        std::to_string(direct_batch_queue_count) +
        " queries=" + std::to_string(ready_queries) + "/" +
        std::to_string(kernel_blocks) +
        " dispatcher=" + std::to_string(ready_dispatchers) + "/1" +
        " control=" + std::to_string(ready_controls) + "/1" +
        " first_owner_phase=" + std::to_string(first_owner_phase));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  kernel_running = true;
  std::cerr << "[gpu-search] unified persistent CTAs=" << owner_kernel_blocks
            << "-owner+" << kernel_blocks
            << "-query+1-dispatch+1-control"
            << " QP-owner-warps=" << direct_batch_queue_count
            << " threads/CTA=" << kPersistentQueryThreads
            << " query_slots=" << query_slots << '\n';
}

void PersistentSearchEngine::Impl::stop_persistent_kernel() {
  if (!kernel_running) return;
  bind_cuda_device("cudaSetDevice(GPU navigation kernel stop)");
  *stop_host = 1;
  check_cuda(cudaMemcpyAsync(stop_device, stop_host, sizeof(u32),
                             cudaMemcpyHostToDevice, rdma_stream),
             "cudaMemcpyAsync(GPU navigation stop)");
  check_cuda(cudaStreamSynchronize(rdma_stream),
             "cudaStreamSynchronize(GPU navigation stop signal)");
  const cudaError_t query_status = cudaStreamSynchronize(kernel_stream);
  const cudaError_t control_status = cudaStreamSynchronize(delta_stream);
  const cudaError_t rdma_status = cudaStreamSynchronize(rdma_stream);
  kernel_running = false;
  check_cuda(query_status, "cudaStreamSynchronize(GPU navigation stop)");
  check_cuda(control_status, "cudaStreamSynchronize(GPU delta control stop)");
  check_cuda(rdma_status, "cudaStreamSynchronize(GPU RDMA owner stop)");
}

PersistentSearchEngine::Impl::~Impl() {
  const cudaError_t device_status = cudaSetDevice(static_cast<int>(config.gpu_device));
  if (device_status != cudaSuccess) {
    std::cerr << "[gpu-search] failed to bind CUDA device during teardown: "
              << cudaGetErrorString(device_status) << '\n';
  }
  accepting.store(false, std::memory_order_release);
  maintenance_shutdown.store(true, std::memory_order_release);
  maintenance_cv.notify_all();
  admission_cv.notify_all();
  slot_cv.notify_all();
  if (maintenance_thread.joinable()) maintenance_thread.join();
  shutdown.store(true, std::memory_order_release);
  admission_cv.notify_all();
  if (admission_thread.joinable()) admission_thread.join();
  reject_queued_submissions("persistent GPU query engine is stopping");
  const auto drain_deadline = std::chrono::steady_clock::now() +
    std::chrono::milliseconds(config.storage_owner_rpc_timeout_ms);
  while (pending_count.load(std::memory_order_acquire) != 0 &&
         std::chrono::steady_clock::now() < drain_deadline) {
    std::this_thread::yield();
  }
  if (kernel_running) {
    *stop_host = 1;
    if (rdma_stream != nullptr) {
      (void)cudaMemcpyAsync(stop_device, stop_host, sizeof(u32),
                            cudaMemcpyHostToDevice, rdma_stream);
      (void)cudaStreamSynchronize(rdma_stream);
    }
    if (kernel_stream != nullptr) cudaStreamSynchronize(kernel_stream);
    if (delta_stream != nullptr) cudaStreamSynchronize(delta_stream);
    if (rdma_stream != nullptr) cudaStreamSynchronize(rdma_stream);
    kernel_running = false;
  }
  reject_all_pending("persistent GPU query engine stopped before completion");
  if (completion_thread.joinable()) completion_thread.join();
  if (route_refresh_stream != nullptr) cudaStreamDestroy(route_refresh_stream);
  if (rdma_stream != nullptr) cudaStreamDestroy(rdma_stream);
  if (delta_stream != nullptr) cudaStreamDestroy(delta_stream);
  if (kernel_stream != nullptr) cudaStreamDestroy(kernel_stream);
  if (direct_disabled_host != nullptr) cudaFreeHost(direct_disabled_host);
  if (direct_error_host != nullptr) cudaFreeHost(direct_error_host);
  if (direct_owner_phases_host != nullptr) cudaFreeHost(direct_owner_phases_host);
  if (control_kernel_ready_host != nullptr) cudaFreeHost(control_kernel_ready_host);
  if (dispatcher_kernel_ready_host != nullptr) {
    cudaFreeHost(dispatcher_kernel_ready_host);
  }
  if (query_kernel_ready_host != nullptr) cudaFreeHost(query_kernel_ready_host);
  if (stop_host != nullptr) cudaFreeHost(stop_host);
  if (result_distances_host != nullptr) cudaFreeHost(result_distances_host);
  if (result_ids_host != nullptr) cudaFreeHost(result_ids_host);
  if (delta_staging_vectors_host != nullptr) cudaFreeHost(delta_staging_vectors_host);
  if (delta_staging_records_host != nullptr) cudaFreeHost(delta_staging_records_host);
  if (delta_staging_slots_host != nullptr) cudaFreeHost(delta_staging_slots_host);
  if (delta_override_updates_host != nullptr) cudaFreeHost(delta_override_updates_host);
  if (delta_durable_updates_host != nullptr) cudaFreeHost(delta_durable_updates_host);
  if (resident_pq_erase_updates_host != nullptr) {
    cudaFreeHost(resident_pq_erase_updates_host);
  }
  if (dynamic_route_updates_host != nullptr) {
    cudaFreeHost(dynamic_route_updates_host);
  }
  if (dynamic_route_code_updates_host != nullptr) {
    cudaFreeHost(dynamic_route_code_updates_host);
  }
  if (delta_supersede_updates_host != nullptr) cudaFreeHost(delta_supersede_updates_host);
  if (graph_invalidation_keys_host != nullptr) cudaFreeHost(graph_invalidation_keys_host);
  if (anchor_graph_validation_host != nullptr) {
    cudaFreeHost(anchor_graph_validation_host);
  }
  if (anchor_graph_readers_host != nullptr) cudaFreeHost(anchor_graph_readers_host);
  device_free(direct_error_device);
  device_free(direct_disabled_device);
  device_free(stop_device);
  device_free(d_dynamic_route_pq_codes);
  device_free(d_dynamic_route_slots);
  device_free(d_delta_count);
  device_free(d_direct_batch_statuses);
  device_free(d_direct_batch_queues);
  device_free(d_direct_batch_entries);
  device_free(d_direct_batch_sequences);
  device_free(d_direct_batch_dequeue);
  device_free(d_direct_batch_enqueue);
  device_free(d_query_dispatch_entries);
  device_free(d_query_dispatch_sequences);
  device_free(d_query_dispatch_dequeue);
  device_free(d_query_dispatch_enqueue);
  device_free(d_resident_pq_positions);
  device_free(d_resident_pq_slots);
  device_free(d_resident_pq_keys);
  device_free(d_resident_pq_codes);
  device_free(d_delta_remote_slots);
  device_free(d_delta_remote_keys);
  device_free(d_base_override_epochs);
  device_free(d_base_override_keys);
  device_free(d_permanent_override_bits);
  device_free(d_delta_remote_positions);
  device_free(d_delta_prev);
  device_free(d_delta_next);
  device_free(d_delta_pq_codes);
  device_free(d_delta_encode_scratch);
  device_free(d_delta_vectors);
  device_free(d_delta_records);
  device_free(d_anchor_graph_readers);
  device_free(d_anchor_graph_states);
  device_free(d_anchor_graph_keys);
  control_bootstrapper.reset();
  if (owns_remote_buffer) device_free(d_remote_buffer);
#ifdef DVSTOR_HAVE_GPUNETIO
  direct_transport.reset();
#endif
  device_free(d_dynamic_code_request_local_iovas);
  device_free(d_dynamic_code_request_offsets);
  device_free(d_dynamic_code_request_shards);
  device_free(d_visited);
  device_free(d_navigation_candidate_distances);
  device_free(d_navigation_candidate_handles);
  device_free(d_query_luts);
  device_free(d_transformed_queries);
  if (query_input_host != nullptr) cudaFreeHost(query_input_host);
  device_free(d_queries);
  device_free(d_delta_bucket_heads);
  device_free(d_anchor_handles);
  device_free(d_anchor_pq_codes);
  device_free(d_anchor_vectors);
  device_free(d_entry_points);
  device_free(d_pq_centroids);
  device_free(d_opq_matrix);
  device_free(d_shards);
}

}  // namespace gpu_search
