#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "common/vector_dtype.hh"
#include "gpu_search/graph_record_validation.hh"
#include "gpu_search/host_distance_kernel.hh"
#include "gpu_search/host_orchestrated_engine/impl.hh"
#include "gpu_search/host_orchestrated_policy.hh"
#include "vamana/storage_layout_resolver.hh"
#include "vamana/vamana_node.hh"

namespace gpu_search {
namespace {

using Clock = std::chrono::steady_clock;
using host_orchestrated_policy::ResolvedRecord;

void atomic_max_query(std::atomic<u64>& value, u64 candidate) {
  u64 observed = value.load(std::memory_order_relaxed);
  while (candidate > observed &&
         !value.compare_exchange_weak(observed, candidate,
                                      std::memory_order_relaxed,
                                      std::memory_order_relaxed)) {
  }
}

template <class Request>
u64 populated_shards(std::span<const Request> requests, u32 shard_count) {
  std::vector<bool> populated(shard_count, false);
  for (const auto& request : requests) {
    if (request.shard < shard_count) populated[request.shard] = true;
  }
  return static_cast<u64>(std::count(populated.begin(), populated.end(), true));
}

}  // namespace

std::vector<HostOrchestratedSearchEngine::Impl::Candidate>
HostOrchestratedSearchEngine::Impl::route_seeds(
  const RouteSnapshot& routes, std::span<const f32> query) const {
  const auto home =
    centroid_home::select_published_snapshot(query, routes.home);
  if (!home.has_value() || *home >= routes.shards.size()) return {};
  const RouteShard& shard = routes.shards[*home];
  std::vector<Candidate> seeds;
  seeds.reserve(shard.live_entry_count);
  for (u32 entry = 0; entry < shard.live_entry_count; ++entry) {
    const auto& route = shard.entries[entry];
    if (route.remote_node == 0 ||
        route.flags != format::kStorageCentroidRouteLive) {
      continue;
    }
    const RemotePtr pointer{route.remote_node};
    if (!pointer.is_well_formed() || pointer.memory_node() != *home) continue;
    seeds.push_back({
      .pointer = pointer,
      .distance = FLT_MAX,
      .expanded = false,
      .extent_class = VamanaNode::DYNAMIC_CODE_EXTENT_CLASS_UNKNOWN,
    });
  }
  return seeds;
}

void HostOrchestratedSearchEngine::Impl::score_candidates(
  Lane& lane, std::vector<Candidate>& candidates) {
  if (candidates.empty()) return;
  if (candidates.size() > score_capacity) {
    throw std::logic_error("host score batch exceeds bounded workspace");
  }
  const auto started = Clock::now();
  std::vector<ReadRequest> dynamic_reads;
  dynamic_reads.reserve(candidates.size());
  std::vector<ResolvedRecord> records(candidates.size());
  std::vector<bool> resolved(candidates.size(), false);
  for (size_t candidate = 0; candidate < candidates.size(); ++candidate) {
    ResolvedRecord record;
    if (!host_orchestrated_policy::resolve_record(
          index, candidates[candidate].pointer, storage_region_bytes, record)) {
      continue;
    }
    if (!record.immutable_base &&
        (record.node_offset > dynamic_allocation_limit ||
         index.shards[record.shard].dynamic_record_bytes >
           dynamic_allocation_limit - record.node_offset)) {
      continue;
    }
    records[candidate] = record;
    resolved[candidate] = true;
    lane.ordinals[candidate] = record.static_ordinal;
    std::memset(lane.packed_dynamic_codes.data() + candidate * code_bytes, 0,
                code_bytes);
    if (!record.immutable_base) {
      dynamic_reads.push_back({
        .shard = record.shard,
        .remote_offset = record.dynamic_code_offset,
        .local_offset =
          dynamic_code_scratch_offset + candidate * dynamic_code_record_stride,
        .bytes = dynamic_code_record_bytes,
      });
    }
  }
  if (!dynamic_reads.empty()) {
    const auto wait_started = Clock::now();
    read_batch(lane, dynamic_reads, true);
    engine.telemetry_.dynamic_code_wait_ns.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                           wait_started)
        .count(),
      std::memory_order_relaxed);
    engine.telemetry_.dynamic_code_reads.fetch_add(dynamic_reads.size(),
                                                   std::memory_order_relaxed);
    engine.telemetry_.dynamic_code_read_bytes.fetch_add(
      static_cast<u64>(dynamic_reads.size()) * dynamic_code_record_bytes,
      std::memory_order_relaxed);
  }

  size_t output = 0;
  for (size_t candidate = 0; candidate < candidates.size(); ++candidate) {
    if (!resolved[candidate]) continue;
    Candidate item = candidates[candidate];
    const ResolvedRecord& record = records[candidate];
    if (!record.immutable_base) {
      const byte_t* dynamic_record = lane.scratch +
                                     dynamic_code_scratch_offset +
                                     candidate * dynamic_code_record_stride;
      u8 extent_class = VamanaNode::DYNAMIC_CODE_EXTENT_CLASS_UNKNOWN;
      if (!host_orchestrated_policy::dynamic_code_snapshot_visible(
            dynamic_record, code_bytes, item.pointer, &extent_class)) {
        engine.telemetry_.dynamic_code_incarnation_rejects.fetch_add(
          1, std::memory_order_relaxed);
        continue;
      }
      item.extent_class = extent_class;
      std::memcpy(lane.packed_dynamic_codes.data() + output * code_bytes,
                  dynamic_record + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES,
                  code_bytes);
      lane.ordinals[output] = std::numeric_limits<u32>::max();
      engine.telemetry_.dynamic_code_candidates.fetch_add(
        1, std::memory_order_relaxed);
    } else {
      lane.ordinals[output] = record.static_ordinal;
      if (output != candidate) {
        std::memset(lane.packed_dynamic_codes.data() + output * code_bytes, 0,
                    code_bytes);
      }
    }
    candidates[output++] = item;
  }
  candidates.resize(output);
  if (candidates.empty()) return;

  const auto cuda_started = Clock::now();
  check_lane_cuda(lane, cudaSetDevice(static_cast<int>(config.gpu_device)),
                  "cudaSetDevice(host PQ score)");
  check_lane_cuda(
    lane,
    cudaMemcpyAsync(lane.d_lut, lane.lut.data(), lane.lut.size() * sizeof(f32),
                    cudaMemcpyHostToDevice, lane.stream),
    "cudaMemcpyAsync(host PQ LUT)");
  check_lane_cuda(lane,
                  cudaMemcpyAsync(lane.d_ordinals, lane.ordinals.data(),
                                  candidates.size() * sizeof(u32),
                                  cudaMemcpyHostToDevice, lane.stream),
                  "cudaMemcpyAsync(host PQ ordinals)");
  check_lane_cuda(
    lane,
    cudaMemcpyAsync(lane.d_dynamic_codes, lane.packed_dynamic_codes.data(),
                    candidates.size() * code_bytes, cudaMemcpyHostToDevice,
                    lane.stream),
    "cudaMemcpyAsync(host dynamic PQ codes)");
  host_distance::launch_pq(lane.stream, d_base_codes, lane.d_ordinals,
                           lane.d_dynamic_codes,
                           static_cast<u32>(candidates.size()), code_bytes,
                           lane.d_lut, lane.d_distances);
  check_lane_cuda(lane, cudaGetLastError(), "host PQ distance launch");
  check_lane_cuda(lane,
                  cudaMemcpyAsync(lane.distances.data(), lane.d_distances,
                                  candidates.size() * sizeof(f32),
                                  cudaMemcpyDeviceToHost, lane.stream),
                  "cudaMemcpyAsync(host PQ distances)");
  check_lane_cuda(lane, cudaStreamSynchronize(lane.stream),
                  "cudaStreamSynchronize(host PQ score)");
  for (size_t candidate = 0; candidate < candidates.size(); ++candidate) {
    candidates[candidate].distance = lane.distances[candidate];
  }
  const u64 elapsed =
    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started)
      .count();
  const u64 cuda_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             Clock::now() - cuda_started)
                             .count();
  engine.telemetry_.gpu_score_ns.fetch_add(elapsed, std::memory_order_relaxed);
  engine.telemetry_.gpu_pq_score_ns.fetch_add(cuda_elapsed,
                                              std::memory_order_relaxed);
  engine.telemetry_.completion_score_batches.fetch_add(
    1, std::memory_order_relaxed);
  engine.telemetry_.completion_score_candidates.fetch_add(
    candidates.size(), std::memory_order_relaxed);
}

void HostOrchestratedSearchEngine::Impl::fetch_graph_wave(
  Lane& lane, std::span<Candidate*> wave,
  std::vector<std::vector<RemotePtr>>& neighbors) {
  neighbors.assign(wave.size(), {});
  if (wave.empty()) return;
  if (wave.size() > config.gpu_graph_commit_width) {
    throw std::logic_error("host graph wave exceeds commit width");
  }
  const auto started = Clock::now();
  struct State {
    ResolvedRecord record{};
    u32 bytes{};
    bool partial{};
    bool started_partial{};
    bool active{};
    bool stale{};
    bool validated{};
  };
  std::vector<State> states(wave.size());
  std::vector<ReadRequest> reads;
  reads.reserve(wave.size());
  const bool adaptive = config.gpu_query_graph_read_policy == "live-extent";
  for (size_t item = 0; item < wave.size(); ++item) {
    Candidate& candidate = *wave[item];
    ResolvedRecord record;
    if (!host_orchestrated_policy::resolve_record(
          index, candidate.pointer, storage_region_bytes, record)) {
      continue;
    }
    if (!record.immutable_base &&
        (record.node_offset > dynamic_allocation_limit ||
         index.shards[record.shard].dynamic_record_bytes >
           dynamic_allocation_limit - record.node_offset)) {
      continue;
    }
    const auto canonical =
      vamana::StorageLayoutResolver::neighbor_read(candidate.pointer).address;
    if (canonical.memory_node != record.shard ||
        canonical.offset != record.graph_offset ||
        canonical.size != graph_entry_bytes) {
      throw std::logic_error(
        "schema-v16 graph resolver disagrees with host query resolver");
    }
    u8 extent_class = graph_record_validation::kGraphExtentClassUnknown;
    if (adaptive && record.immutable_base) {
      extent_class = graph_extent_classes[record.static_ordinal];
    } else if (adaptive && config.gpu_dynamic_graph_extent &&
               candidate.extent_class !=
                 VamanaNode::DYNAMIC_CODE_EXTENT_CLASS_UNKNOWN) {
      extent_class = candidate.extent_class;
    }
    const u32 bytes =
      adaptive ? graph_record_validation::graph_extent_bytes_for_class(
                   extent_class, graph_entry_bytes, graph_entry_capacity)
               : graph_entry_bytes;
    const size_t local = graph_scratch_offset + item * graph_entry_bytes;
    std::memset(lane.scratch + local, 0, graph_entry_bytes);
    states[item] = {
      .record = record,
      .bytes = bytes,
      .partial = bytes < graph_entry_bytes,
      .started_partial = bytes < graph_entry_bytes,
      .active = true,
    };
    reads.push_back({
      .shard = record.shard,
      .remote_offset = record.graph_offset,
      .local_offset = local,
      .bytes = bytes,
    });
    engine.telemetry_.graph_read_bytes.fetch_add(bytes,
                                                 std::memory_order_relaxed);
    engine.telemetry_.critical_graph_bytes.fetch_add(bytes,
                                                     std::memory_order_relaxed);
    if (!record.immutable_base) {
      engine.telemetry_.dynamic_graph_read_bytes.fetch_add(
        bytes, std::memory_order_relaxed);
      (states[item].partial ? engine.telemetry_.dynamic_graph_short_reads
                            : engine.telemetry_.dynamic_graph_full_reads)
        .fetch_add(1, std::memory_order_relaxed);
    }
    (states[item].partial ? engine.telemetry_.graph_live_extent_reads
                          : engine.telemetry_.graph_full_record_reads)
      .fetch_add(1, std::memory_order_relaxed);
  }
  read_batch(lane, reads, true);
  engine.telemetry_.graph_page_requests.fetch_add(reads.size(),
                                                  std::memory_order_relaxed);
  engine.telemetry_.critical_graph_reads.fetch_add(reads.size(),
                                                   std::memory_order_relaxed);
  engine.telemetry_.graph_shard_batches.fetch_add(
    populated_shards<ReadRequest>(reads, index.shards.size()),
    std::memory_order_relaxed);
  engine.telemetry_.critical_misses.fetch_add(reads.size(),
                                              std::memory_order_relaxed);

  // A checksum failure is normally a read racing a whole-record graph
  // publication, not persistent corruption.  Under the coupled baseline,
  // hundreds of host query lanes can repeatedly hit the same centroid-near
  // parents while backlink writers update them.  Three back-to-back reads
  // were too small a contention budget and made one ordinary race terminate
  // an otherwise healthy long benchmark.  Match the persistent reader's
  // accounting rule (an optimistic short extent is outside the authoritative
  // full-record budget) and give the CPU-posted path enough bounded full
  // rereads to outlive a hot publication window.  The final invalid full
  // snapshot still fails loudly, so durable corruption is never accepted.
  constexpr u32 kFullGraphSnapshotAttempts = 8;
  const u32 maximum_batch_attempts =
    adaptive ? kFullGraphSnapshotAttempts + 1u : kFullGraphSnapshotAttempts;
  u64 validation_ns = 0;
  for (u32 attempt = 0; attempt < maximum_batch_attempts; ++attempt) {
    const auto validation_started = Clock::now();
    std::vector<ReadRequest> retry;
    for (size_t item = 0; item < wave.size(); ++item) {
      State& state = states[item];
      if (!state.active || state.stale) continue;
      const byte_t* record =
        lane.scratch + graph_scratch_offset + item * graph_entry_bytes;
      const auto snapshot =
        state.partial
          ? graph_record_validation::classify_zero_extended_snapshot(
              record, state.bytes, graph_entry_bytes, config.R,
              graph_entry_capacity, wave[item]->pointer.incarnation())
          : graph_record_validation::classify_snapshot(
              record, graph_entry_bytes, config.R, graph_entry_capacity,
              wave[item]->pointer.incarnation());
      if (snapshot == graph_record_validation::SnapshotState::valid) {
        u32 required_bytes = 0;
        if (adaptive && graph_record_validation::required_live_extent_bytes(
                          record, state.bytes, config.R, graph_entry_capacity,
                          required_bytes)) {
          const u8 required_class =
            graph_record_validation::graph_extent_class_for_required_bytes(
              required_bytes, graph_entry_capacity);
          Candidate& candidate = *wave[item];
          if (!state.record.immutable_base && config.gpu_dynamic_graph_extent &&
              required_class !=
                graph_record_validation::kGraphExtentClassUnknown) {
            const u8 previous = candidate.extent_class;
            if (previous != VamanaNode::DYNAMIC_CODE_EXTENT_CLASS_UNKNOWN) {
              if (required_class > previous) {
                engine.telemetry_.dynamic_graph_hint_promotions.fetch_add(
                  1, std::memory_order_relaxed);
              } else if (required_class < previous) {
                engine.telemetry_.dynamic_graph_hint_demotions.fetch_add(
                  1, std::memory_order_relaxed);
              }
            }
            candidate.extent_class = required_class;
          } else if (state.record.immutable_base && state.started_partial &&
                     !state.partial) {
            engine.telemetry_.graph_extent_hint_promotions.fetch_add(
              1, std::memory_order_relaxed);
          }
        }
        state.validated = true;
        state.active = false;
        continue;
      }
      if (snapshot ==
          graph_record_validation::SnapshotState::stale_incarnation) {
        state.stale = true;
        state.active = false;
        continue;
      }
      const bool attempts_remain =
        graph_record_validation::snapshot_retry_available(
          attempt, state.started_partial ? 1u : 0u, state.partial,
          maximum_batch_attempts, kFullGraphSnapshotAttempts);
      if (!attempts_remain) {
        throw std::runtime_error(
          "host query graph snapshot validation failed after bounded "
          "full-record rereads");
      }
      if (state.partial) {
        engine.telemetry_.graph_extent_fallback_reads.fetch_add(
          1, std::memory_order_relaxed);
        engine.telemetry_.graph_extent_underhint_reads.fetch_add(
          1, std::memory_order_relaxed);
        if (!state.record.immutable_base) {
          engine.telemetry_.dynamic_graph_fallback_reads.fetch_add(
            1, std::memory_order_relaxed);
        }
      } else {
        engine.telemetry_.graph_read_retries.fetch_add(
          1, std::memory_order_relaxed);
      }
      state.partial = false;
      state.bytes = graph_entry_bytes;
      const size_t local = graph_scratch_offset + item * graph_entry_bytes;
      std::memset(lane.scratch + local, 0, graph_entry_bytes);
      retry.push_back({
        .shard = state.record.shard,
        .remote_offset = state.record.graph_offset,
        .local_offset = local,
        .bytes = graph_entry_bytes,
      });
      engine.telemetry_.graph_read_bytes.fetch_add(graph_entry_bytes,
                                                   std::memory_order_relaxed);
      engine.telemetry_.critical_graph_bytes.fetch_add(
        graph_entry_bytes, std::memory_order_relaxed);
      engine.telemetry_.graph_full_record_reads.fetch_add(
        1, std::memory_order_relaxed);
      if (!state.record.immutable_base) {
        engine.telemetry_.dynamic_graph_read_bytes.fetch_add(
          graph_entry_bytes, std::memory_order_relaxed);
        engine.telemetry_.dynamic_graph_full_reads.fetch_add(
          1, std::memory_order_relaxed);
      }
    }
    validation_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                       Clock::now() - validation_started)
                       .count();
    if (retry.empty()) break;
    read_batch(lane, retry, true);
    engine.telemetry_.graph_shard_batches.fetch_add(
      populated_shards<ReadRequest>(retry, index.shards.size()),
      std::memory_order_relaxed);
    engine.telemetry_.graph_page_requests.fetch_add(retry.size(),
                                                    std::memory_order_relaxed);
    engine.telemetry_.critical_graph_reads.fetch_add(retry.size(),
                                                     std::memory_order_relaxed);
  }
  engine.telemetry_.gpu_graph_validation_ns.fetch_add(
    validation_ns, std::memory_order_relaxed);

  const auto decode_started = Clock::now();
  std::vector<byte_t> decoded(VamanaNode::neighbor_read_size());
  for (size_t item = 0; item < wave.size(); ++item) {
    if (host_orchestrated_policy::graph_snapshot_decodable(
          states[item].validated, states[item].stale)) {
      const byte_t* record =
        lane.scratch + graph_scratch_offset + item * graph_entry_bytes;
      if (!VamanaNode::decode_hot_graph_entry(
            record, decoded.data(), wave[item]->pointer.incarnation())) {
        throw std::runtime_error("host query graph decode failed");
      }
      const u32 count = VamanaNode::decoded_neighbor_count(decoded.data());
      const auto* pointers = reinterpret_cast<const RemotePtr*>(
        decoded.data() + VamanaNode::neighbor_payload_offset_in_read());
      neighbors[item].assign(pointers, pointers + count);
    }
  }
  engine.telemetry_.gpu_neighbor_decode_ns.fetch_add(
    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                         decode_started)
      .count(),
    std::memory_order_relaxed);
  const u64 elapsed =
    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started)
      .count();
  engine.telemetry_.gpu_graph_ns.fetch_add(elapsed, std::memory_order_relaxed);
}

service::QueryResult HostOrchestratedSearchEngine::Impl::exact_rerank(
  Lane& lane, std::span<const Candidate> beam, u32 k) {
  const auto started = Clock::now();
  const size_t attempted = std::min<size_t>(beam.size(), exact_capacity);
  std::vector<ReadRequest> records;
  std::vector<RemotePtr> handles;
  records.reserve(attempted);
  handles.reserve(attempted);
  for (size_t candidate = 0; candidate < attempted; ++candidate) {
    ResolvedRecord record;
    if (!host_orchestrated_policy::resolve_record(
          index, beam[candidate].pointer, storage_region_bytes, record)) {
      continue;
    }
    if (!record.immutable_base &&
        (record.node_offset > dynamic_allocation_limit ||
         index.shards[record.shard].dynamic_record_bytes >
           dynamic_allocation_limit - record.node_offset)) {
      continue;
    }
    const size_t output = handles.size();
    const size_t local = exact_scratch_offset + output * exact_record_stride;
    std::memset(lane.scratch + local, 0, exact_record_stride);
    records.push_back({
      .shard = record.shard,
      .remote_offset = record.node_offset,
      .local_offset = local,
      .bytes = exact_record_bytes,
    });
    handles.push_back(beam[candidate].pointer);
  }
  read_batch(lane, records, true);
  std::vector<ReadRequest> headers(records.size());
  for (size_t record = 0; record < records.size(); ++record) {
    headers[record] = {
      .shard = records[record].shard,
      .remote_offset = records[record].remote_offset,
      .local_offset = exact_header_scratch_offset + record * sizeof(u64),
      .bytes = sizeof(u64),
    };
  }
  read_batch(lane, headers, true);
  const u64 exact_shard_batches =
    populated_shards<ReadRequest>(records, index.shards.size());
  engine.telemetry_.exact_snapshot_train_batches.fetch_add(
    exact_shard_batches, std::memory_order_relaxed);
  // A CPU-owned QP expresses the same snapshot dependency as two bounded
  // completion batches (payload, then current header), not the GPUNetIO
  // fenced-tail train. Report that distinction instead of claiming the fast
  // persistent transport path.
  engine.telemetry_.exact_snapshot_train_fallbacks.fetch_add(
    exact_shard_batches, std::memory_order_relaxed);
  engine.telemetry_.exact_vector_reads.fetch_add(records.size(),
                                                 std::memory_order_relaxed);

  std::vector<RemotePtr> valid_handles;
  std::vector<node_t> ids;
  valid_handles.reserve(records.size());
  ids.reserve(records.size());
  size_t valid = 0;
  for (size_t record = 0; record < records.size(); ++record) {
    u64 header_after = 0;
    std::memcpy(
      &header_after,
      lane.scratch + exact_header_scratch_offset + record * sizeof(u64),
      sizeof(header_after));
    const byte_t* source =
      lane.scratch + exact_scratch_offset + record * exact_record_stride;
    if (!host_orchestrated_policy::exact_snapshot_visible(
          source, exact_record_bytes, header_after, handles[record])) {
      continue;
    }
    if (valid != record) {
      std::memmove(
        lane.scratch + exact_scratch_offset + valid * exact_record_stride,
        source, exact_record_stride);
      source =
        lane.scratch + exact_scratch_offset + valid * exact_record_stride;
    }
    node_t id = 0;
    std::memcpy(&id, source + VamanaNode::offset_id(), sizeof(id));
    ids.push_back(id);
    valid_handles.push_back(handles[record]);
    ++valid;
  }
  if (valid == 0) return {};

  check_lane_cuda(lane, cudaSetDevice(static_cast<int>(config.gpu_device)),
                  "cudaSetDevice(host exact score)");
  check_lane_cuda(lane,
                  cudaMemcpyAsync(lane.d_query, lane.query.data(),
                                  lane.query.size() * sizeof(f32),
                                  cudaMemcpyHostToDevice, lane.stream),
                  "cudaMemcpyAsync(host exact query)");
  check_lane_cuda(
    lane,
    cudaMemcpyAsync(lane.d_exact_records, lane.scratch + exact_scratch_offset,
                    valid * exact_record_stride, cudaMemcpyHostToDevice,
                    lane.stream),
    "cudaMemcpyAsync(host exact records)");
  host_distance::launch_exact(
    lane.stream, lane.d_query, lane.d_exact_records, static_cast<u32>(valid),
    config.dim, config.resolved_vector_dtype(), exact_record_stride,
    static_cast<u32>(VamanaNode::offset_vector()), lane.d_exact_distances);
  check_lane_cuda(lane, cudaGetLastError(), "host exact distance launch");
  check_lane_cuda(
    lane,
    cudaMemcpyAsync(lane.distances.data(), lane.d_exact_distances,
                    valid * sizeof(f32), cudaMemcpyDeviceToHost, lane.stream),
    "cudaMemcpyAsync(host exact distances)");
  check_lane_cuda(lane, cudaStreamSynchronize(lane.stream),
                  "cudaStreamSynchronize(host exact score)");

  struct Exact {
    node_t id{};
    f32 distance{};
    RemotePtr pointer{};
  };
  std::vector<Exact> exact(valid);
  for (size_t record = 0; record < valid; ++record) {
    exact[record] = {ids[record], lane.distances[record],
                     valid_handles[record]};
  }
  std::sort(exact.begin(), exact.end(),
            [](const Exact& left, const Exact& right) {
              if (left.distance != right.distance) {
                return left.distance < right.distance;
              }
              if (left.id != right.id) return left.id < right.id;
              return left.pointer.raw_address < right.pointer.raw_address;
            });
  service::QueryResult result;
  result.reserve(std::min<size_t>(k, exact.size()));
  for (size_t item = 0; item < exact.size() && item < k; ++item) {
    result.push_back({exact[item].id, exact[item].distance});
  }
  engine.telemetry_.gpu_exact_ns.fetch_add(
    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started)
      .count(),
    std::memory_order_relaxed);
  return result;
}

service::QueryResult HostOrchestratedSearchEngine::Impl::execute_query(
  Lane& lane, VectorDType query_dtype, const byte_t* query_data, u32 k,
  const std::shared_ptr<const RouteSnapshot>& routes) {
  const auto prepare_started = Clock::now();
  decode_storage_vector_to_float(query_data, query_dtype, config.dim,
                                 lane.query.data());
  for (f32 value : lane.query) {
    if (!floating_value_is_finite(value)) {
      throw std::invalid_argument("host query components must be finite");
    }
  }
  pq::build_distance_table(pq_model, lane.query, lane.lut, lane.transformed);
  engine.telemetry_.gpu_prepare_ns.fetch_add(
    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                         prepare_started)
      .count(),
    std::memory_order_relaxed);
  lane.beam = route_seeds(*routes, lane.query);
  if (!lane.beam.empty()) {
    engine.telemetry_.graph_route_hits.fetch_add(1, std::memory_order_relaxed);
  }
  score_candidates(lane, lane.beam);
  if (lane.beam.empty()) return {};
  const auto less = [](const Candidate& left, const Candidate& right) {
    return host_orchestrated_policy::distance_handle_less(
      left.distance, left.pointer, right.distance, right.pointer);
  };
  std::sort(lane.beam.begin(), lane.beam.end(), less);
  if (lane.beam.size() > config.gpu_traversal_beam_width) {
    lane.beam.resize(config.gpu_traversal_beam_width);
  }
  lane.visited.clear();
  for (const Candidate& seed : lane.beam) {
    lane.visited.insert(seed.pointer.raw_address);
  }

  u32 expansions = 0;
  while (expansions < config.gpu_max_expansions) {
    const auto selection_started = Clock::now();
    std::vector<Candidate*> wave;
    wave.reserve(config.gpu_graph_commit_width);
    for (Candidate& candidate : lane.beam) {
      if (candidate.expanded) continue;
      candidate.expanded = true;
      wave.push_back(&candidate);
      if (wave.size() == config.gpu_graph_commit_width ||
          expansions + wave.size() == config.gpu_max_expansions) {
        break;
      }
    }
    if (wave.empty()) break;
    const auto selection_complete = Clock::now();
    const u64 selection_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(selection_complete -
                                                           selection_started)
        .count();
    engine.telemetry_.gpu_beam_selection_ns.fetch_add(
      selection_ns, std::memory_order_relaxed);
    engine.telemetry_.gpu_beam_ns.fetch_add(selection_ns,
                                            std::memory_order_relaxed);
    std::vector<std::vector<RemotePtr>> adjacency;
    fetch_graph_wave(lane, wave, adjacency);
    expansions += static_cast<u32>(wave.size());
    const auto visited_started = Clock::now();
    lane.pending.clear();
    for (const auto& neighbors : adjacency) {
      for (RemotePtr neighbor : neighbors) {
        if (!neighbor.is_well_formed() || neighbor.is_null() ||
            neighbor.memory_node() >= index.shards.size() ||
            !lane.visited.insert(neighbor.raw_address).second) {
          continue;
        }
        lane.pending.push_back({
          .pointer = neighbor,
          .distance = FLT_MAX,
          .expanded = false,
          .extent_class = VamanaNode::DYNAMIC_CODE_EXTENT_CLASS_UNKNOWN,
        });
      }
    }
    engine.telemetry_.gpu_visited_ns.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                           visited_started)
        .count(),
      std::memory_order_relaxed);
    score_candidates(lane, lane.pending);
    const auto merge_started = Clock::now();
    lane.beam.insert(lane.beam.end(), lane.pending.begin(), lane.pending.end());
    std::sort(lane.beam.begin(), lane.beam.end(), less);
    lane.beam.erase(
      std::unique(lane.beam.begin(), lane.beam.end(),
                  [](const Candidate& left, const Candidate& right) {
                    return left.pointer == right.pointer;
                  }),
      lane.beam.end());
    if (lane.beam.size() > config.gpu_traversal_beam_width) {
      lane.beam.resize(config.gpu_traversal_beam_width);
    }
    const u64 merge_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           Clock::now() - merge_started)
                           .count();
    engine.telemetry_.gpu_beam_merge_ns.fetch_add(merge_ns,
                                                  std::memory_order_relaxed);
    engine.telemetry_.gpu_beam_ns.fetch_add(merge_ns,
                                            std::memory_order_relaxed);
    engine.telemetry_.graph_dependency_rounds.fetch_add(
      1, std::memory_order_relaxed);
    engine.telemetry_.issue_epochs.fetch_add(1, std::memory_order_relaxed);
    engine.telemetry_.commit_epochs.fetch_add(1, std::memory_order_relaxed);
    engine.telemetry_.issue_width_sum.fetch_add(wave.size(),
                                                std::memory_order_relaxed);
    engine.telemetry_.issue_width_capacity_sum.fetch_add(
      config.gpu_graph_commit_width, std::memory_order_relaxed);
    engine.telemetry_.commit_width_sum.fetch_add(wave.size(),
                                                 std::memory_order_relaxed);
    engine.telemetry_.logical_expansions.fetch_add(wave.size(),
                                                   std::memory_order_relaxed);
    atomic_max_query(engine.telemetry_.max_commit_width, wave.size());
    atomic_max_query(engine.telemetry_.max_issue_width, wave.size());
  }
  return exact_rerank(lane, lane.beam, k);
}

service::QueryResult HostOrchestratedSearchEngine::Impl::search(
  VectorDType query_dtype, const byte_t* query_data, u32 k) {
  if (stopping.load(std::memory_order_acquire)) {
    throw std::runtime_error(unhealthy_message());
  }
  if (query_data == nullptr ||
      static_cast<u32>(query_dtype) > static_cast<u32>(VectorDType::int8) ||
      k == 0 || k > std::max(config.k, config.gpu_final_rerank_width)) {
    throw std::invalid_argument("invalid host-orchestrated query");
  }
  LaneGuard guard = acquire_lane(true);
  Lane& lane = guard.get();
  engine.telemetry_.queries_submitted.fetch_add(1, std::memory_order_relaxed);
  const auto query_started = Clock::now();
  for (u32 route_attempt = 0; route_attempt < 2; ++route_attempt) {
    auto routes =
      std::atomic_load_explicit(&route_snapshot, std::memory_order_acquire);
    if (routes == nullptr) {
      throw std::runtime_error("host centroid routes are unavailable");
    }
    service::QueryResult result =
      execute_query(lane, query_dtype, query_data, k, routes);
    if (!result.empty()) {
      if (!healthy.load(std::memory_order_acquire) ||
          stopping.load(std::memory_order_acquire)) {
        throw std::runtime_error(unhealthy_message());
      }
      engine.telemetry_.queries_completed.fetch_add(1,
                                                    std::memory_order_relaxed);
      engine.telemetry_.batches.fetch_add(1, std::memory_order_relaxed);
      engine.telemetry_.batch_queries.fetch_add(1, std::memory_order_relaxed);
      engine.telemetry_.gpu_active_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                             query_started)
          .count(),
        std::memory_order_relaxed);
      return result;
    }
    engine.telemetry_.centroid_route_query_retries.fetch_add(
      1, std::memory_order_relaxed);
    std::lock_guard refresh_lock(route_refresh_mutex);
    engine.telemetry_.graph_route_refreshes.fetch_add(
      1, std::memory_order_relaxed);
    (void)synchronize_storage_routes(lane);
  }
  engine.telemetry_.centroid_route_query_timeouts.fetch_add(
    1, std::memory_order_relaxed);
  throw std::runtime_error(
    "host query produced no exact-visible result after route refresh");
}

}  // namespace gpu_search
