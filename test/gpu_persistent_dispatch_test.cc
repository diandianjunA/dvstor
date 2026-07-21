#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "gpu_search/mapped_ring.hh"
#include "gpu_search/persistent_kernel.hh"
#include "remote_pointer.hh"

namespace {

using gpu_search::MappedRing;
using gpu_search::f32;
using gpu_search::u32;
using gpu_search::u64;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <typename T>
T* device_allocate(size_t count, const char* operation) {
  T* result = nullptr;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&result), count * sizeof(T)),
             operation);
  return result;
}

template <typename T>
void copy_to_device(T* destination, const T* source, size_t count,
                    const char* operation) {
  check_cuda(cudaMemcpy(destination, source, count * sizeof(T),
                        cudaMemcpyHostToDevice), operation);
}

template <typename T>
void copy_from_device(T* destination, const T* source, size_t count,
                      const char* operation) {
  check_cuda(cudaMemcpy(destination, source, count * sizeof(T),
                        cudaMemcpyDeviceToHost), operation);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count == 0) {
      std::cout << "SKIP: no CUDA device available\n";
      return 0;
    }
    check_cuda(cudaSetDevice(0), "cudaSetDevice");

    MappedRing<gpu_search::QueryDescriptor> submissions(
      256, MappedRing<gpu_search::QueryDescriptor>::Direction::host_to_device);
    MappedRing<gpu_search::CompletionDescriptor> completions(
      256, MappedRing<gpu_search::CompletionDescriptor>::Direction::device_to_host);
    MappedRing<gpu_search::CentroidRoutePublishDescriptor> route_submissions(
      8,
      MappedRing<gpu_search::CentroidRoutePublishDescriptor>::Direction::
        host_to_device);
    MappedRing<gpu_search::CentroidRoutePublishCompletion> route_completions(
      8,
      MappedRing<gpu_search::CentroidRoutePublishCompletion>::Direction::
        device_to_host);

    constexpr u32 kShardCount = 2;
    constexpr u32 kDim = 4;
    constexpr u32 kEntryCapacity =
      gpu_search::kCentroidRouteMaxLiveEntries;
    auto* route_updates = device_allocate<gpu_search::CentroidRouteUpdate>(
      kShardCount, "cudaMalloc(route updates)");
    auto* centroid_updates = device_allocate<f32>(
      kShardCount * kDim, "cudaMalloc(centroid updates)");
    auto* route_shards =
      device_allocate<gpu_search::DeviceCentroidRouteShard>(
        kShardCount, "cudaMalloc(route shards)");
    auto* route_entries =
      device_allocate<gpu_search::DeviceCentroidRouteEntry>(
        kShardCount * kEntryCapacity, "cudaMalloc(route entries)");
    auto* shard_centroids = device_allocate<f32>(
      kShardCount * kDim, "cudaMalloc(shard centroids)");
    auto* route_epoch = device_allocate<u64>(
      1, "cudaMalloc(route publication epoch)");
    check_cuda(cudaMemset(route_shards, 0,
                          kShardCount * sizeof(*route_shards)),
               "cudaMemset(route shards)");
    check_cuda(cudaMemset(route_entries, 0,
                          kShardCount * kEntryCapacity *
                            sizeof(*route_entries)),
               "cudaMemset(route entries)");
    check_cuda(cudaMemset(shard_centroids, 0,
                          kShardCount * kDim * sizeof(*shard_centroids)),
               "cudaMemset(shard centroids)");
    check_cuda(cudaMemset(route_epoch, 0, sizeof(*route_epoch)),
               "cudaMemset(route publication epoch)");

    u32* stop_host = nullptr;
    u32* stop_device = nullptr;
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&stop_host), sizeof(u32),
                             cudaHostAllocMapped),
               "cudaHostAlloc(stop)");
    *stop_host = 0;
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&stop_device),
                                        stop_host, 0),
               "cudaHostGetDevicePointer(stop)");

    gpu_search::PersistentKernelParams params{
      .submissions = submissions.device_view(),
      .device_submissions = {},
      .completions = completions.device_view(),
      .route_submissions = route_submissions.device_view(),
      .route_completions = route_completions.device_view(),
      .num_shards = kShardCount,
      .dim = kDim,
      .query_slots = 1,
      .centroid_route_updates = route_updates,
      .centroid_route_centroid_updates = centroid_updates,
      .centroid_route_shards = route_shards,
      .centroid_route_entries = route_entries,
      .shard_centroids = shard_centroids,
      .centroid_route_epoch = route_epoch,
      .centroid_route_shard_capacity = kShardCount,
      .centroid_route_entry_capacity = kEntryCapacity,
      .stop = stop_device,
    };

    const u32 blocks = argc > 3
      ? static_cast<u32>(std::max(1, std::atoi(argv[3]))) : 8;
    const u32 threads = argc > 5
      ? static_cast<u32>(std::max(32, std::atoi(argv[5]))) : 128;
    if (threads != 128 && threads != 256) {
      throw std::invalid_argument("query threads must be 128 or 256");
    }
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
    gpu_search::launch_persistent_search(stream, params, blocks, threads);
    check_cuda(cudaPeekAtLastError(), "launch_persistent_search");

    const int timeout_seconds = argc > 2
      ? std::max(1, std::atoi(argv[2])) : 10;
    const auto deadline_after = [&] {
      return std::chrono::steady_clock::now() +
        std::chrono::seconds(timeout_seconds);
    };
    const auto publish = [&](u64 command_id, u32 update_count) {
      const auto deadline = deadline_after();
      const gpu_search::CentroidRoutePublishDescriptor descriptor{
        .command_id = command_id,
        .update_count = update_count,
      };
      while (!route_submissions.try_push(descriptor)) {
        if (std::chrono::steady_clock::now() >= deadline) {
          throw std::runtime_error("centroid route submission stalled");
        }
        std::this_thread::yield();
      }
      gpu_search::CentroidRoutePublishCompletion completion{};
      while (!route_completions.try_pop(completion)) {
        if (std::chrono::steady_clock::now() >= deadline) {
          throw std::runtime_error("centroid route publication did not complete");
        }
        std::this_thread::yield();
      }
      return completion;
    };

    std::array<gpu_search::CentroidRouteUpdate, kShardCount> updates{};
    updates[0] = {
      .version = 11,
      .vector_count = 10,
      .shard = 0,
      .live_entry_count = 1,
    };
    updates[0].entries[0] = {
      .remote_node = RemotePtr{0, 0x100}.raw_address,
      .generation = 3,
      .flags = gpu_search::kCentroidRouteLive,
    };
    updates[1] = {
      .version = 7,
      .vector_count = 5,
      .shard = 1,
      .live_entry_count = 2,
    };
    updates[1].entries[0] = {
      .remote_node = RemotePtr{1, 0x200}.raw_address,
      .generation = 4,
      .flags = gpu_search::kCentroidRouteLive,
    };
    updates[1].entries[1] = {
      .remote_node = RemotePtr{1, 0x300}.raw_address,
      .generation = 5,
      .flags = gpu_search::kCentroidRouteLive,
    };
    const std::array<f32, kShardCount * kDim> centroids{
      1.0f, 2.0f, 3.0f, 4.0f,
      -1.0f, -2.0f, -3.0f, -4.0f,
    };
    copy_to_device(route_updates, updates.data(), updates.size(),
                   "cudaMemcpy(route updates)");
    copy_to_device(centroid_updates, centroids.data(), centroids.size(),
                   "cudaMemcpy(centroid updates)");
    const auto installed = publish(41, kShardCount);
    if (installed.command_id != 41 || installed.status != 0 ||
        installed.update_count != kShardCount) {
      throw std::runtime_error("valid centroid route publication was rejected");
    }

    std::array<gpu_search::DeviceCentroidRouteShard, kShardCount> shard_state{};
    std::array<gpu_search::DeviceCentroidRouteEntry,
               kShardCount * kEntryCapacity> entry_state{};
    std::array<f32, kShardCount * kDim> centroid_state{};
    u64 epoch_state = 0;
    copy_from_device(shard_state.data(), route_shards, shard_state.size(),
                     "cudaMemcpy(route shard state)");
    copy_from_device(entry_state.data(), route_entries, entry_state.size(),
                     "cudaMemcpy(route entry state)");
    copy_from_device(centroid_state.data(), shard_centroids,
                     centroid_state.size(), "cudaMemcpy(centroid state)");
    copy_from_device(&epoch_state, route_epoch, 1,
                     "cudaMemcpy(route publication epoch)");
    if (shard_state[0].sequence != 2 || shard_state[0].command_id != 41 ||
        shard_state[0].version != 11 || shard_state[0].vector_count != 10 ||
        shard_state[0].live_entry_count != 1 ||
        shard_state[1].sequence != 2 || shard_state[1].command_id != 41 ||
        shard_state[1].version != 7 || shard_state[1].vector_count != 5 ||
        shard_state[1].live_entry_count != 2 ||
        entry_state[0].remote_node != RemotePtr{0, 0x100}.raw_address ||
        entry_state[kEntryCapacity].remote_node !=
          RemotePtr{1, 0x200}.raw_address ||
        entry_state[kEntryCapacity + 1].remote_node !=
          RemotePtr{1, 0x300}.raw_address ||
        centroid_state != centroids || epoch_state != 2) {
      throw std::runtime_error("centroid route transaction was torn");
    }

    // Reusing a command id is stale and must leave the installed transaction
    // unchanged.
    const auto stale = publish(41, kShardCount);
    if (stale.command_id != 41 || stale.status != -ESTALE ||
        stale.update_count != 0) {
      throw std::runtime_error("stale centroid route publication was accepted");
    }
    copy_from_device(&epoch_state, route_epoch, 1,
                     "cudaMemcpy(stale route publication epoch)");
    if (epoch_state != 2) {
      throw std::runtime_error("rejected route publication changed its epoch");
    }

    // A zero-count shard transaction is a first-class route withdrawal. It
    // clears every old entry under the same seqlock as the centroid update.
    updates[0] = {
      .version = 12,
      .vector_count = 0,
      .shard = 0,
      .live_entry_count = 0,
    };
    const std::array<f32, kDim> empty_centroid{9.0f, 8.0f, 7.0f, 6.0f};
    copy_to_device(route_updates, updates.data(), 1,
                   "cudaMemcpy(empty route update)");
    copy_to_device(centroid_updates, empty_centroid.data(),
                   empty_centroid.size(), "cudaMemcpy(empty centroid)");
    const auto withdrawn = publish(42, 1);
    if (withdrawn.command_id != 42 || withdrawn.status != 0 ||
        withdrawn.update_count != 1) {
      throw std::runtime_error("centroid route withdrawal failed");
    }
    copy_from_device(shard_state.data(), route_shards, shard_state.size(),
                     "cudaMemcpy(withdrawn shard state)");
    copy_from_device(entry_state.data(), route_entries, entry_state.size(),
                     "cudaMemcpy(withdrawn entry state)");
    copy_from_device(centroid_state.data(), shard_centroids,
                     centroid_state.size(), "cudaMemcpy(withdrawn centroid)");
    copy_from_device(&epoch_state, route_epoch, 1,
                     "cudaMemcpy(withdrawn route publication epoch)");
    if (shard_state[0].sequence != 4 || shard_state[0].command_id != 42 ||
        shard_state[0].version != 12 || shard_state[0].vector_count != 0 ||
        shard_state[0].live_entry_count != 0 ||
        !std::equal(empty_centroid.begin(), empty_centroid.end(),
                    centroid_state.begin()) || epoch_state != 4) {
      throw std::runtime_error("centroid route withdrawal was torn");
    }
    for (u32 entry = 0; entry < kEntryCapacity; ++entry) {
      if (entry_state[entry].remote_node != 0 ||
          entry_state[entry].generation != 0 ||
          entry_state[entry].flags != 0) {
        throw std::runtime_error("centroid route withdrawal retained an entry");
      }
    }

    constexpr u64 kRequestBase = 0x1020304050600000ULL;
    constexpr u32 kBatchSize = 64;
    const u32 query_count = argc > 1
      ? static_cast<u32>(std::max(1, std::atoi(argv[1]))) : 1024;
    const auto query_deadline = deadline_after();
    for (u32 batch_begin = 0; batch_begin < query_count;
         batch_begin += kBatchSize) {
      const u32 batch_size = std::min(kBatchSize, query_count - batch_begin);
      for (u32 index = 0; index < batch_size; ++index) {
        const gpu_search::QueryDescriptor query{
          .request_id = kRequestBase + batch_begin + index,
          .query_slot = 0,
          .result_capacity = 10,
          .dim = 0,
          .k = 10,
        };
        while (!submissions.try_push(query)) {
          if (std::chrono::steady_clock::now() >= query_deadline) {
            throw std::runtime_error("persistent submission ring stalled");
          }
          std::this_thread::yield();
        }
      }

      std::array<bool, kBatchSize> seen{};
      for (u32 completed = 0; completed < batch_size;) {
        gpu_search::CompletionDescriptor completion{};
        if (!completions.try_pop(completion)) {
          if (std::chrono::steady_clock::now() >= query_deadline) {
            throw std::runtime_error("persistent query did not complete");
          }
          std::this_thread::yield();
          continue;
        }
        const u64 first_request = kRequestBase + batch_begin;
        if (completion.request_id < first_request ||
            completion.request_id >= first_request + batch_size ||
            completion.query_slot != 0 || completion.status != -EINVAL) {
          throw std::runtime_error("persistent completion is invalid");
        }
        const u32 index = static_cast<u32>(completion.request_id - first_request);
        if (seen[index]) {
          throw std::runtime_error("persistent query completed more than once");
        }
        seen[index] = true;
        ++completed;
      }
    }

    std::atomic_ref<u32>(*stop_host).store(1, std::memory_order_release);
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    check_cuda(cudaFreeHost(stop_host), "cudaFreeHost(stop)");
    check_cuda(cudaFree(route_epoch), "cudaFree(route publication epoch)");
    check_cuda(cudaFree(shard_centroids), "cudaFree(shard centroids)");
    check_cuda(cudaFree(route_entries), "cudaFree(route entries)");
    check_cuda(cudaFree(route_shards), "cudaFree(route shards)");
    check_cuda(cudaFree(centroid_updates), "cudaFree(centroid updates)");
    check_cuda(cudaFree(route_updates), "cudaFree(route updates)");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
