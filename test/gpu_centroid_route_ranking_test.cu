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

#include "gpu_search/persistent_kernel/query_traversal.cuh"

namespace {

using gpu_search::DeviceCentroidRouteEntry;
using gpu_search::DeviceCentroidRouteShard;
using gpu_search::PersistentKernelParams;
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
