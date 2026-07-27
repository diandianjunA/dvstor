#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_search/persistent_kernel/query_traversal.cuh"

namespace {

using gpu_search::DeviceCentroidRouteEntry;
using gpu_search::DeviceCentroidRouteShard;
using gpu_search::PersistentKernelParams;
using gpu_search::CompletionDescriptor;
using gpu_search::QueryDescriptor;
using gpu_search::f32;
using gpu_search::u32;
using gpu_search::u64;
using gpu_search::centroid_route_ranking::RankedShard;

constexpr u32 kShards = gpu_search::kPersistentMaxShards;
constexpr u32 kDim = 257;
constexpr u32 kEntries = gpu_search::kCentroidRouteMaxLiveEntries;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <typename T>
T* device_copy(const T* source, std::size_t count) {
  T* destination = nullptr;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&destination),
                        count * sizeof(T)), "cudaMalloc");
  check_cuda(cudaMemcpy(destination, source, count * sizeof(T),
                        cudaMemcpyHostToDevice), "cudaMemcpy H2D");
  return destination;
}

__global__ void snapshot_and_rank_kernel(PersistentKernelParams params,
                                         const f32* query,
                                         RankedShard* output) {
  __shared__ RankedShard routes[kShards];
  for (u32 shard = threadIdx.x; shard < kShards; shard += blockDim.x) {
    routes[shard] = RankedShard{
      .distance = FLT_MAX, .shard = shard, .valid = 0};
    gpu_search::persistent_kernel_detail::CentroidRouteShardSnapshot snapshot;
    if (shard < params.num_shards &&
        gpu_search::persistent_kernel_detail::snapshot_centroid_route_shard(
          params, shard, query, snapshot)) {
      routes[shard] = RankedShard{
        .distance = snapshot.distance, .shard = shard, .valid = 1};
    }
  }
  __syncthreads();
  gpu_search::persistent_kernel_detail::sort_centroid_route_shards(routes);
  for (u32 index = threadIdx.x; index < kShards; index += blockDim.x) {
    output[index] = routes[index];
  }
}

__global__ void classify_graph_record_kernel(
    const std::uint8_t* record, u32 record_bytes,
    u32 expected_incarnation, u32* output) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const auto state =
    gpu_search::graph_record_validation::classify_snapshot(
      record, record_bytes, 3, 5, expected_incarnation);
  const auto action =
    gpu_search::graph_record_validation::decide_read_action(
      true, state, false);
  output[0] = static_cast<u32>(state);
  output[1] = static_cast<u32>(action);
}

__global__ void promote_graph_extent_hint_kernel(
    PersistentKernelParams params,
    u64 handle,
    u32 required_bytes,
    u32* output) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  output[0] = gpu_search::persistent_kernel_detail::
    promote_graph_extent_class(params, handle, required_bytes) ? 1u : 0u;
  u32 ordinal = 0;
  output[1] =
    gpu_search::persistent_kernel_detail::static_ordinal_from_raw(
      params, handle, ordinal)
      ? gpu_search::persistent_kernel_detail::load_graph_extent_class(
          params, ordinal)
      : gpu_search::graph_record_validation::kGraphExtentClassUnknown;
}

__global__ void concurrently_promote_graph_extent_hint_kernel(
    PersistentKernelParams params,
    u64 handle,
    u32* output) {
  const u32 requested_class = 3u + threadIdx.x % 11u;
  const u32 live_edges = (requested_class - 1u) * 8u + 1u;
  const u32 required_bytes =
    gpu_search::graph_record_validation::kGraphRecordHeaderBytes +
      live_edges * sizeof(u64);
  if (gpu_search::persistent_kernel_detail::promote_graph_extent_class(
        params, handle, required_bytes)) {
    atomicAdd(output, 1u);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    u32 ordinal = 0;
    output[1] =
      gpu_search::persistent_kernel_detail::static_ordinal_from_raw(
        params, handle, ordinal)
        ? gpu_search::persistent_kernel_detail::load_graph_extent_class(
            params, ordinal)
        : gpu_search::graph_record_validation::kGraphExtentClassUnknown;
  }
}

__global__ void route_contention_query_kernel(PersistentKernelParams params,
                                               QueryDescriptor descriptor) {
  gpu_search::persistent_kernel_detail::process_query(params, descriptor);
}

__global__ void close_route_epoch_after(u64* epoch, u64 delay_ns) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  u64 started = 0;
  u64 now = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(started));
  do {
    __nanosleep(256);
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  } while (now - started < delay_ns);
  atomicExch(reinterpret_cast<unsigned long long*>(epoch), 2ULL);
  __threadfence();
}

void test_route_publication_contention() {
  constexpr u32 kRouteDim = 1;
  constexpr u32 kRequestCapacity = gpu_search::kPersistentMaxMergeCandidates;

  std::vector<void*> allocations;
  const auto allocate = [&](std::size_t bytes) {
    void* result = nullptr;
    check_cuda(cudaMalloc(&result, bytes), "cudaMalloc(route contention state)");
    allocations.push_back(result);
    check_cuda(cudaMemset(result, 0, bytes), "cudaMemset(route contention state)");
    return result;
  };

  auto* shard = static_cast<gpu_search::DeviceShardRegion*>(
    allocate(sizeof(gpu_search::DeviceShardRegion)));
  const gpu_search::DeviceShardRegion shard_host{
    .ordinal_base = 0,
    .node_count = 1,
    .node_base_offset = 0x1000,
    .node_stride = 64,
    .graph_base_offset = 0x2000,
    .dynamic_base_offset = 0x4000,
    .memory_node = 0,
    .dynamic_record_bytes = 64,
    .dynamic_hot_offset = 16,
    .dynamic_code_offset = 32,
  };
  check_cuda(cudaMemcpy(shard, &shard_host, sizeof(shard_host),
                        cudaMemcpyHostToDevice), "cudaMemcpy(route shard)");

  auto* route_shard = static_cast<DeviceCentroidRouteShard*>(
    allocate(sizeof(DeviceCentroidRouteShard)));
  const DeviceCentroidRouteShard route_shard_host{
    .sequence = 2,
    .command_id = 1,
    .version = 1,
    .vector_count = 1,
    .live_entry_count = 1,
  };
  check_cuda(cudaMemcpy(route_shard, &route_shard_host,
                        sizeof(route_shard_host), cudaMemcpyHostToDevice),
             "cudaMemcpy(route shard header)");
  auto* route_entries = static_cast<DeviceCentroidRouteEntry*>(
    allocate(gpu_search::kCentroidRouteMaxLiveEntries *
             sizeof(DeviceCentroidRouteEntry)));
  const DeviceCentroidRouteEntry route_entry{
    .remote_node = 0x1000 >> 4,
    .generation = 1,
    .flags = gpu_search::kCentroidRouteLive,
  };
  check_cuda(cudaMemcpy(route_entries, &route_entry, sizeof(route_entry),
                        cudaMemcpyHostToDevice), "cudaMemcpy(route entry)");
  auto* route_epoch = static_cast<u64*>(allocate(sizeof(u64)));
  const u64 odd_epoch = 1;
  check_cuda(cudaMemcpy(route_epoch, &odd_epoch, sizeof(odd_epoch),
                        cudaMemcpyHostToDevice), "cudaMemcpy(odd route epoch)");

  auto* query_input = static_cast<std::uint8_t*>(allocate(sizeof(std::uint8_t)));
  auto* result_id = static_cast<u32*>(allocate(sizeof(u32)));
  auto* result_distance = static_cast<f32*>(allocate(sizeof(f32)));
  auto* pq_code = static_cast<std::uint8_t*>(allocate(sizeof(std::uint8_t)));
  auto* pq_centroids = static_cast<f32*>(allocate(256 * sizeof(f32)));
  auto* decoded_query = static_cast<f32*>(allocate(sizeof(f32)));
  auto* transformed_query = static_cast<f32*>(allocate(sizeof(f32)));
  auto* query_lut = static_cast<f32*>(allocate(256 * sizeof(f32)));
  auto* navigation_handles = static_cast<u64*>(
    allocate(kRequestCapacity * sizeof(u64)));
  auto* navigation_distances = static_cast<f32*>(
    allocate(kRequestCapacity * sizeof(f32)));
  auto* visited = static_cast<u64*>(allocate(256 * sizeof(u64)));
  auto* exact_records = static_cast<std::uint8_t*>(allocate(24));
  auto* request_shards = static_cast<u32*>(
    allocate(kRequestCapacity * sizeof(u32)));
  auto* request_offsets = static_cast<u64*>(
    allocate(kRequestCapacity * sizeof(u64)));
  auto* request_iovas = static_cast<u64*>(
    allocate(kRequestCapacity * sizeof(u64)));
  auto* stop = static_cast<u32*>(allocate(sizeof(u32)));
  auto* shard_centroid = static_cast<f32*>(allocate(sizeof(f32)));
  auto* completion_enqueue = static_cast<unsigned long long*>(
    allocate(sizeof(unsigned long long)));
  auto* completion_dequeue = static_cast<unsigned long long*>(
    allocate(sizeof(unsigned long long)));
  auto* completion_sequences = static_cast<unsigned long long*>(
    allocate(2 * sizeof(unsigned long long)));
  const std::array<unsigned long long, 2> initial_sequences{0, 1};
  check_cuda(cudaMemcpy(completion_sequences, initial_sequences.data(),
                        sizeof(initial_sequences), cudaMemcpyHostToDevice),
             "cudaMemcpy(completion sequences)");
  auto* completion_entries = static_cast<CompletionDescriptor*>(
    allocate(2 * sizeof(CompletionDescriptor)));

  PersistentKernelParams params{
    .completions = {
      .enqueue_position = completion_enqueue,
      .dequeue_position = completion_dequeue,
      .sequences = completion_sequences,
      .entries = completion_entries,
      .capacity = 2,
      .mask = 1,
    },
    .shards = shard,
    .num_shards = 1,
    .pq_codes = pq_code,
    .pq_centroids = pq_centroids,
    .num_nodes = 1,
    .dim = kRouteDim,
    .pq_subquantizers = 1,
    .pq_subvector_dim = 1,
    .pq_code_bytes = 1,
    .graph_entry_bytes = 16,
    .graph_degree = 1,
    .graph_entry_capacity = 1,
    .node_record_bytes = 16,
    .node_record_stride = 24,
    .node_vector_offset = 8,
    .node_incarnation_offset = 4,
    .vector_bytes = 1,
    .vector_dtype = 1,
    .traversal_beam_width = 1,
    .final_rerank_width = 1,
    .exact_width = 1,
    .max_expansions = 0,
    .visited_capacity = 256,
    .query_slots = 1,
    .route_snapshot_timeout_ns = 100'000'000ULL,
    .centroid_route_shards = route_shard,
    .centroid_route_entries = route_entries,
    .shard_centroids = shard_centroid,
    .centroid_route_epoch = route_epoch,
    .centroid_route_shard_capacity = 1,
    .centroid_route_entry_capacity =
      gpu_search::kCentroidRouteMaxLiveEntries,
    .stop = stop,
    .decoded_queries = decoded_query,
    .transformed_queries = transformed_query,
    .query_luts = query_lut,
    .navigation_candidate_handles = navigation_handles,
    .navigation_candidate_distances = navigation_distances,
    .visited_hash = visited,
    .exact_records = exact_records,
    .dynamic_code_request_shards = request_shards,
    .dynamic_code_request_offsets = request_offsets,
    .dynamic_code_request_local_iovas = request_iovas,
    .result_ids = result_id,
    .result_distances = result_distance,
  };
  const QueryDescriptor descriptor{
    .request_id = 7,
    .query_device_address = reinterpret_cast<u64>(query_input),
    .result_device_address = reinterpret_cast<u64>(result_id),
    .query_slot = 0,
    .result_capacity = 1,
    .dim = 1,
    .k = 1,
    .query_dtype = 1,
  };

  cudaStream_t query_stream = nullptr;
  cudaStream_t writer_stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&query_stream, cudaStreamNonBlocking),
             "cudaStreamCreate(query contention)");
  check_cuda(cudaStreamCreateWithFlags(&writer_stream, cudaStreamNonBlocking),
             "cudaStreamCreate(route writer)");
  route_contention_query_kernel<<<1, 128, 0, query_stream>>>(params, descriptor);
  check_cuda(cudaGetLastError(), "route_contention_query_kernel launch");
  close_route_epoch_after<<<1, 1, 0, writer_stream>>>(route_epoch, 5'000'000ULL);
  check_cuda(cudaGetLastError(), "close_route_epoch_after launch");
  check_cuda(cudaStreamSynchronize(query_stream),
             "cudaStreamSynchronize(route contention query)");
  check_cuda(cudaStreamSynchronize(writer_stream),
             "cudaStreamSynchronize(route writer)");

  CompletionDescriptor completion{};
  check_cuda(cudaMemcpy(&completion, completion_entries, sizeof(completion),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(route contention completion)");
  assert(completion.request_id == descriptor.request_id);
  assert(completion.status == -EIO);
  assert(gpu_search::query_failure_reason(completion.diagnostic) ==
         gpu_search::QueryFailureReason::exact_rerank_empty);
  assert(gpu_search::query_route_snapshot_retries(completion.diagnostic) != 0);

  check_cuda(cudaStreamDestroy(writer_stream), "cudaStreamDestroy(route writer)");
  check_cuda(cudaStreamDestroy(query_stream), "cudaStreamDestroy(query contention)");
  for (void* allocation : allocations) check_cuda(cudaFree(allocation), "cudaFree");
}

void test_graph_extent_hint_promotion() {
  constexpr u32 graph_capacity = 102;
  const gpu_search::DeviceShardRegion shard{
    .ordinal_base = 0,
    .node_count = 3,
    .node_base_offset = 0x1000,
    .node_stride = 64,
    .memory_node = 0,
  };
  auto* device_shard = device_copy(&shard, 1);
  const u32 initial_word =
    static_cast<u32>(
      gpu_search::graph_record_validation::kGraphExtentClassUnknown) |
      (2u << 8) | (1u << 16) | (0x5au << 24);
  auto* device_word = device_copy(&initial_word, 1);
  u32* device_output = nullptr;
  check_cuda(cudaMalloc(
    reinterpret_cast<void**>(&device_output), 2 * sizeof(u32)),
    "cudaMalloc(extent promotion output)");

  PersistentKernelParams params{};
  params.shards = device_shard;
  params.num_shards = 1;
  params.num_nodes = 3;
  params.graph_entry_capacity = graph_capacity;
  params.graph_extent_class_words = device_word;
  const u64 ordinal_one = (u64{0x1000} + 64) >> 4;
  const u32 class_five_bytes =
    gpu_search::graph_record_validation::kGraphRecordHeaderBytes +
      33 * sizeof(u64);
  promote_graph_extent_hint_kernel<<<1, 32>>>(
    params, ordinal_one, class_five_bytes, device_output);
  check_cuda(cudaGetLastError(), "promote_graph_extent_hint_kernel launch");
  check_cuda(cudaDeviceSynchronize(),
             "promote_graph_extent_hint_kernel sync");
  std::array<u32, 2> output{};
  check_cuda(cudaMemcpy(
    output.data(), device_output, sizeof(output), cudaMemcpyDeviceToHost),
    "cudaMemcpy(extent promotion output)");
  assert(output[0] == 1);
  assert(output[1] == 5);

  u32 promoted_word = 0;
  check_cuda(cudaMemcpy(
    &promoted_word, device_word, sizeof(promoted_word),
    cudaMemcpyDeviceToHost), "cudaMemcpy(promoted extent word)");
  assert(gpu_search::graph_record_validation::packed_graph_extent_class(
           promoted_word, 0) ==
         gpu_search::graph_record_validation::kGraphExtentClassUnknown);
  assert(gpu_search::graph_record_validation::packed_graph_extent_class(
           promoted_word, 1) == 5);
  assert(gpu_search::graph_record_validation::packed_graph_extent_class(
           promoted_word, 2) == 1);
  assert(gpu_search::graph_record_validation::packed_graph_extent_class(
           promoted_word, 3) == 0x5a);

  // A lower request cannot regress a learned class, and an unknown/full byte
  // remains authoritative rather than being replaced by a short-read hint.
  promote_graph_extent_hint_kernel<<<1, 32>>>(
    params, ordinal_one,
    gpu_search::graph_record_validation::kGraphRecordHeaderBytes +
      24 * sizeof(u64),
    device_output);
  check_cuda(cudaGetLastError(), "nonregressing extent promotion launch");
  check_cuda(cudaDeviceSynchronize(), "nonregressing extent promotion sync");
  check_cuda(cudaMemcpy(
    output.data(), device_output, sizeof(output), cudaMemcpyDeviceToHost),
    "cudaMemcpy(nonregressing extent promotion output)");
  assert(output[0] == 0);
  assert(output[1] == 5);

  const u64 ordinal_zero = u64{0x1000} >> 4;
  promote_graph_extent_hint_kernel<<<1, 32>>>(
    params, ordinal_zero, class_five_bytes, device_output);
  check_cuda(cudaGetLastError(), "unknown extent promotion launch");
  check_cuda(cudaDeviceSynchronize(), "unknown extent promotion sync");
  check_cuda(cudaMemcpy(
    output.data(), device_output, sizeof(output), cudaMemcpyDeviceToHost),
    "cudaMemcpy(unknown extent promotion output)");
  assert(output[0] == 0);
  assert(output[1] ==
         gpu_search::graph_record_validation::kGraphExtentClassUnknown);

  // The third byte is the final real ordinal in a partially occupied device
  // word. Its CAS must remain in bounds and preserve the padding byte.
  const u64 ordinal_two = (u64{0x1000} + 2 * 64) >> 4;
  promote_graph_extent_hint_kernel<<<1, 32>>>(
    params, ordinal_two, class_five_bytes, device_output);
  check_cuda(cudaGetLastError(), "partial-word extent promotion launch");
  check_cuda(cudaDeviceSynchronize(), "partial-word extent promotion sync");
  check_cuda(cudaMemcpy(
    &promoted_word, device_word, sizeof(promoted_word),
    cudaMemcpyDeviceToHost), "cudaMemcpy(partial-word extent word)");
  assert(gpu_search::graph_record_validation::packed_graph_extent_class(
           promoted_word, 2) == 5);
  assert(gpu_search::graph_record_validation::packed_graph_extent_class(
           promoted_word, 3) == 0x5a);

  // Multiple CTAs/threads may observe the same stale byte. Every successful
  // transition is monotonic and the maximum requested class must win without
  // corrupting adjacent ordinals.
  check_cuda(cudaMemset(device_output, 0, 2 * sizeof(u32)),
             "cudaMemset(concurrent extent output)");
  concurrently_promote_graph_extent_hint_kernel<<<1, 32>>>(
    params, ordinal_one, device_output);
  check_cuda(cudaGetLastError(), "concurrent extent promotion launch");
  check_cuda(cudaDeviceSynchronize(), "concurrent extent promotion sync");
  check_cuda(cudaMemcpy(
    output.data(), device_output, sizeof(output), cudaMemcpyDeviceToHost),
    "cudaMemcpy(concurrent extent promotion output)");
  assert(output[0] >= 1);
  assert(output[0] <= 8);
  assert(output[1] == 13);
  check_cuda(cudaMemcpy(
    &promoted_word, device_word, sizeof(promoted_word),
    cudaMemcpyDeviceToHost), "cudaMemcpy(concurrent extent word)");
  assert(gpu_search::graph_record_validation::packed_graph_extent_class(
           promoted_word, 0) ==
         gpu_search::graph_record_validation::kGraphExtentClassUnknown);
  assert(gpu_search::graph_record_validation::packed_graph_extent_class(
           promoted_word, 1) == 13);
  assert(gpu_search::graph_record_validation::packed_graph_extent_class(
           promoted_word, 2) == 5);
  assert(gpu_search::graph_record_validation::packed_graph_extent_class(
           promoted_word, 3) == 0x5a);

  check_cuda(cudaFree(device_output), "cudaFree(extent promotion output)");
  check_cuda(cudaFree(device_word), "cudaFree(extent class word)");
  check_cuda(cudaFree(device_shard), "cudaFree(extent promotion shard)");
}

}  // namespace

int main() {
  try {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count == 0) {
      std::cout << "SKIP: no CUDA device available\n";
      return 0;
    }
    check_cuda(cudaSetDevice(0), "cudaSetDevice");

    std::mt19937 generator(0x1234abcdu);
    std::uniform_real_distribution<f32> component(-32.0f, 32.0f);
    std::array<f32, kDim> query{};
    std::array<f32, static_cast<std::size_t>(kShards) * kDim> centroids{};
    for (f32& value : query) value = component(generator);
    for (f32& value : centroids) value = component(generator);
    // Exact centroid ties must retain physical-shard order.
    std::copy_n(centroids.begin(), kDim, centroids.begin() + 31 * kDim);

    std::array<DeviceCentroidRouteShard, kShards> shards{};
    std::array<DeviceCentroidRouteEntry,
               static_cast<std::size_t>(kShards) * kEntries> entries{};
    std::array<RankedShard, kShards> expected{};
    for (u32 shard = 0; shard < kShards; ++shard) {
      const bool valid = shard != 7 && shard != 29 && shard != 52;
      shards[shard].sequence = shard == 7 ? 3 : 2;
      shards[shard].version = valid ? 10 + shard : 0;
      shards[shard].vector_count = valid ? 100 + shard : 0;
      shards[shard].live_entry_count = valid ? 1 : 0;
      entries[static_cast<std::size_t>(shard) * kEntries] = {
        .remote_node = (static_cast<u64>(shard) <<
                          gpu_search::kRemoteOffsetUnitBits) | 0x10u,
        .generation = 1,
        .flags = gpu_search::kCentroidRouteLive,
      };
      f32 distance = 0.0f;
      for (u32 dimension = 0; dimension < kDim; ++dimension) {
        const f32 difference = query[dimension] -
          centroids[static_cast<std::size_t>(shard) * kDim + dimension];
        distance = std::fma(difference, difference, distance);
      }
      expected[shard] = RankedShard{
        .distance = valid ? distance : FLT_MAX,
        .shard = shard,
        .valid = static_cast<u32>(valid),
      };
    }
    std::sort(expected.begin(), expected.end(),
              gpu_search::centroid_route_ranking::less);

    DeviceCentroidRouteShard* device_shards =
      device_copy(shards.data(), shards.size());
    DeviceCentroidRouteEntry* device_entries =
      device_copy(entries.data(), entries.size());
    f32* device_centroids = device_copy(centroids.data(), centroids.size());
    f32* device_query = device_copy(query.data(), query.size());
    RankedShard* device_output = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_output),
                          sizeof(RankedShard) * kShards),
               "cudaMalloc(output)");

    PersistentKernelParams params{};
    params.num_shards = kShards;
    params.dim = kDim;
    params.centroid_route_shards = device_shards;
    params.centroid_route_entries = device_entries;
    params.shard_centroids = device_centroids;
    params.centroid_route_shard_capacity = kShards;
    params.centroid_route_entry_capacity = kEntries;
    snapshot_and_rank_kernel<<<1, 256>>>(params, device_query, device_output);
    check_cuda(cudaGetLastError(), "snapshot_and_rank_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "snapshot_and_rank_kernel sync");

    std::array<RankedShard, kShards> actual{};
    check_cuda(cudaMemcpy(actual.data(), device_output,
                          actual.size() * sizeof(RankedShard),
                          cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    for (u32 rank = 0; rank < kShards; ++rank) {
      assert(actual[rank].valid == expected[rank].valid);
      assert(actual[rank].shard == expected[rank].shard);
      assert(actual[rank].distance == expected[rank].distance);
    }

    test_route_publication_contention();
    test_graph_extent_hint_promotion();

    // Exercise the same host/device classifier used by graph RDMA reads. A
    // checksum-valid replacement incarnation must discard only the stale
    // candidate, even on the final snapshot attempt.
    std::array<std::uint8_t, 56> graph_record{};
    graph_record[0] = 1;
    graph_record[8] = 12;
    const std::uint16_t graph_checksum =
      gpu_search::graph_record_validation::checksum16(
        graph_record.data(), graph_record.size());
    graph_record[2] = static_cast<std::uint8_t>(graph_checksum);
    graph_record[3] = static_cast<std::uint8_t>(graph_checksum >> 8);
    std::uint8_t* device_graph_record =
      device_copy(graph_record.data(), graph_record.size());
    u32* device_graph_result = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_graph_result),
                          2 * sizeof(u32)),
               "cudaMalloc(graph classification result)");
    classify_graph_record_kernel<<<1, 32>>>(
      device_graph_record, graph_record.size(), 11, device_graph_result);
    check_cuda(cudaGetLastError(), "classify_graph_record_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "classify_graph_record_kernel sync");
    std::array<u32, 2> graph_result{};
    check_cuda(cudaMemcpy(graph_result.data(), device_graph_result,
                          graph_result.size() * sizeof(u32),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(graph classification D2H)");
    assert(graph_result[0] == static_cast<u32>(
      gpu_search::graph_record_validation::SnapshotState::stale_incarnation));
    assert(graph_result[1] == static_cast<u32>(
      gpu_search::graph_record_validation::ReadAction::discard_stale));
    check_cuda(cudaFree(device_graph_result),
               "cudaFree(graph classification result)");
    check_cuda(cudaFree(device_graph_record), "cudaFree(graph record)");

    check_cuda(cudaFree(device_output), "cudaFree(output)");
    check_cuda(cudaFree(device_query), "cudaFree(query)");
    check_cuda(cudaFree(device_centroids), "cudaFree(centroids)");
    check_cuda(cudaFree(device_entries), "cudaFree(entries)");
    check_cuda(cudaFree(device_shards), "cudaFree(shards)");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
