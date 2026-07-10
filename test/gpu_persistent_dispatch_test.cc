#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "gpu_search/persistent_kernel.hh"

namespace {

using gpu_search::u32;
using gpu_search::u64;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <class T>
class MappedRing {
public:
  enum class Direction {
    host_to_device,
    device_to_host,
  };

  MappedRing(u32 capacity, Direction direction)
      : capacity_(capacity), direction_(direction) {
    if (capacity_ < 2 || (capacity_ & (capacity_ - 1)) != 0) {
      throw std::invalid_argument("mapped ring capacity must be a power of two");
    }
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&enqueue_host_), sizeof(u64),
                             cudaHostAllocMapped),
               "cudaHostAlloc(enqueue)");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&dequeue_host_), sizeof(u64),
                             cudaHostAllocMapped),
               "cudaHostAlloc(dequeue)");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&sequences_host_),
                             static_cast<size_t>(capacity_) * sizeof(u64),
                             cudaHostAllocMapped),
               "cudaHostAlloc(sequences)");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&entries_host_),
                             static_cast<size_t>(capacity_) * sizeof(T),
                             cudaHostAllocMapped),
               "cudaHostAlloc(entries)");

    *enqueue_host_ = 0;
    *dequeue_host_ = 0;
    for (u32 index = 0; index < capacity_; ++index) sequences_host_[index] = index;

    u64* enqueue_device = nullptr;
    u64* dequeue_device = nullptr;
    u64* sequences_device = nullptr;
    T* entries_device = nullptr;
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&enqueue_device),
                                        enqueue_host_, 0),
               "cudaHostGetDevicePointer(enqueue)");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dequeue_device),
                                        dequeue_host_, 0),
               "cudaHostGetDevicePointer(dequeue)");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&sequences_device),
                                        sequences_host_, 0),
               "cudaHostGetDevicePointer(sequences)");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&entries_device),
                                        entries_host_, 0),
               "cudaHostGetDevicePointer(entries)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_owned_position_), sizeof(u64)),
               "cudaMalloc(device position)");
    check_cuda(cudaMemset(device_owned_position_, 0, sizeof(u64)),
               "cudaMemset(device position)");
    if (direction_ == Direction::host_to_device) {
      dequeue_device = device_owned_position_;
    } else {
      enqueue_device = device_owned_position_;
    }
    device_view_ = {
      .enqueue_position = reinterpret_cast<unsigned long long*>(enqueue_device),
      .dequeue_position = reinterpret_cast<unsigned long long*>(dequeue_device),
      .sequences = reinterpret_cast<unsigned long long*>(sequences_device),
      .entries = entries_device,
      .capacity = capacity_,
      .mask = capacity_ - 1,
    };
  }

  ~MappedRing() {
    if (device_owned_position_ != nullptr) cudaFree(device_owned_position_);
    if (entries_host_ != nullptr) cudaFreeHost(entries_host_);
    if (sequences_host_ != nullptr) cudaFreeHost(sequences_host_);
    if (dequeue_host_ != nullptr) cudaFreeHost(dequeue_host_);
    if (enqueue_host_ != nullptr) cudaFreeHost(enqueue_host_);
  }

  MappedRing(const MappedRing&) = delete;
  MappedRing& operator=(const MappedRing&) = delete;

  bool try_push(const T& value) {
    std::atomic_ref<u64> enqueue(*enqueue_host_);
    const u64 position = enqueue.load(std::memory_order_relaxed);
    const u32 slot = static_cast<u32>(position) & (capacity_ - 1);
    std::atomic_ref<u64> sequence(sequences_host_[slot]);
    if (sequence.load(std::memory_order_acquire) != position) return false;
    entries_host_[slot] = value;
    sequence.store(position + 1, std::memory_order_release);
    enqueue.store(position + 1, std::memory_order_release);
    return true;
  }

  bool try_pop(T& value) {
    std::atomic_ref<u64> dequeue(*dequeue_host_);
    const u64 position = dequeue.load(std::memory_order_relaxed);
    const u32 slot = static_cast<u32>(position) & (capacity_ - 1);
    std::atomic_ref<u64> sequence(sequences_host_[slot]);
    if (sequence.load(std::memory_order_acquire) != position + 1) return false;
    value = entries_host_[slot];
    sequence.store(position + capacity_, std::memory_order_release);
    dequeue.store(position + 1, std::memory_order_release);
    return true;
  }

  gpu_search::DeviceRingView<T> device_view() const { return device_view_; }

private:
  u32 capacity_{};
  u64* enqueue_host_{};
  u64* dequeue_host_{};
  u64* sequences_host_{};
  T* entries_host_{};
  u64* device_owned_position_{};
  Direction direction_{};
  gpu_search::DeviceRingView<T> device_view_{};
};

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

    u32* stop_host = nullptr;
    u32* stop_device = nullptr;
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&stop_host), sizeof(u32),
                             cudaHostAllocMapped),
               "cudaHostAlloc(stop)");
    *stop_host = 0;
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&stop_device),
                                        stop_host, 0),
               "cudaHostGetDevicePointer(stop)");

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");

    gpu_search::PersistentKernelParams params{};
    params.submissions = submissions.device_view();
    params.completions = completions.device_view();
    params.stop = stop_device;
    params.query_slots = 1;
    params.dim = 128;
    const u32 block_count = argc > 3
      ? static_cast<u32>(std::max(1, std::atoi(argv[3]))) : 8;
    gpu_search::launch_persistent_search(stream, params, block_count, 128);
    check_cuda(cudaPeekAtLastError(), "launch_persistent_search");

    constexpr u64 kRequestBase = 0x1020304050600000ULL;
    constexpr u32 kBatchSize = 64;
    const u32 query_count = argc > 1
      ? static_cast<u32>(std::max(1, std::atoi(argv[1]))) : 1024;
    const int timeout_seconds = argc > 2 ? std::max(1, std::atoi(argv[2])) : 10;
    const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::seconds(timeout_seconds);
    for (u32 batch_begin = 0; batch_begin < query_count; batch_begin += kBatchSize) {
      const u32 batch_size = std::min(kBatchSize, query_count - batch_begin);
      for (u32 index = 0; index < batch_size; ++index) {
        gpu_search::QueryDescriptor query{
          .request_id = kRequestBase + batch_begin + index,
          .snapshot_epoch = 17,
          .query_slot = 0,
          .result_capacity = 10,
          .dim = 0,
          .k = 10,
        };
        while (!submissions.try_push(query)) {
          if (std::chrono::steady_clock::now() >= deadline) {
            std::cerr << "persistent submission ring stopped making progress\n";
            std::_Exit(EXIT_FAILURE);
          }
          std::this_thread::yield();
        }
      }

      bool seen[kBatchSize]{};
      for (u32 completed = 0; completed < batch_size;) {
        gpu_search::CompletionDescriptor completion{};
        if (!completions.try_pop(completion)) {
          if (std::chrono::steady_clock::now() >= deadline) {
            std::cerr << "persistent kernel did not complete CTA-owned queries\n";
            std::_Exit(EXIT_FAILURE);
          }
          std::this_thread::yield();
          continue;
        }
        const u64 first_request = kRequestBase + batch_begin;
        if (completion.request_id < first_request ||
            completion.request_id >= first_request + batch_size ||
            completion.snapshot_epoch != 17 || completion.query_slot != 0 ||
            completion.status != -EINVAL) {
          std::cerr << "unexpected completion: request=" << completion.request_id
                    << " epoch=" << completion.snapshot_epoch
                    << " slot=" << completion.query_slot
                    << " status=" << completion.status << '\n';
          std::_Exit(EXIT_FAILURE);
        }
        const u32 index = static_cast<u32>(completion.request_id - first_request);
        if (seen[index]) {
          std::cerr << "duplicate persistent completion\n";
          std::_Exit(EXIT_FAILURE);
        }
        seen[index] = true;
        ++completed;
      }
    }

    std::atomic_ref<u32>(*stop_host).store(1, std::memory_order_release);
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    constexpr u32 kCacheWays = 4;
    const u64 cache_keys_host[kCacheWays]{11, 22, 33, 44};
    u32 cache_states_host[kCacheWays]{2, 2, 2, 2};
    const u32 cache_readers_host[kCacheWays]{};
    const u64 invalidation_key_host = 22;
    u64* cache_keys_device = nullptr;
    u32* cache_states_device = nullptr;
    u32* cache_readers_device = nullptr;
    u64* invalidation_key_device = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&cache_keys_device),
                          sizeof(cache_keys_host)), "cudaMalloc(cache keys)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&cache_states_device),
                          sizeof(cache_states_host)), "cudaMalloc(cache states)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&cache_readers_device),
                          sizeof(cache_readers_host)), "cudaMalloc(cache readers)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&invalidation_key_device),
                          sizeof(invalidation_key_host)), "cudaMalloc(invalidation key)");
    check_cuda(cudaMemcpy(cache_keys_device, cache_keys_host, sizeof(cache_keys_host),
                          cudaMemcpyHostToDevice), "cudaMemcpy(cache keys)");
    check_cuda(cudaMemcpy(cache_states_device, cache_states_host, sizeof(cache_states_host),
                          cudaMemcpyHostToDevice), "cudaMemcpy(cache states)");
    check_cuda(cudaMemcpy(cache_readers_device, cache_readers_host, sizeof(cache_readers_host),
                          cudaMemcpyHostToDevice), "cudaMemcpy(cache readers)");
    check_cuda(cudaMemcpy(invalidation_key_device, &invalidation_key_host,
                          sizeof(invalidation_key_host), cudaMemcpyHostToDevice),
               "cudaMemcpy(invalidation key)");
    gpu_search::launch_invalidate_graph_cache(
      stream, invalidation_key_device, 1, cache_keys_device, cache_states_device,
      cache_readers_device, 1, kCacheWays);
    check_cuda(cudaGetLastError(), "launch_invalidate_graph_cache");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(cache invalidation)");
    check_cuda(cudaMemcpy(cache_states_host, cache_states_device, sizeof(cache_states_host),
                          cudaMemcpyDeviceToHost), "cudaMemcpy(cache states result)");
    if (cache_states_host[0] != 2 || cache_states_host[1] != 0 ||
        cache_states_host[2] != 2 || cache_states_host[3] != 2) {
      std::cerr << "targeted graph-cache invalidation cleared the wrong cache line\n";
      std::_Exit(EXIT_FAILURE);
    }
    check_cuda(cudaFree(invalidation_key_device), "cudaFree(invalidation key)");
    check_cuda(cudaFree(cache_readers_device), "cudaFree(cache readers)");
    check_cuda(cudaFree(cache_states_device), "cudaFree(cache states)");
    check_cuda(cudaFree(cache_keys_device), "cudaFree(cache keys)");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    check_cuda(cudaFreeHost(stop_host), "cudaFreeHost(stop)");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
