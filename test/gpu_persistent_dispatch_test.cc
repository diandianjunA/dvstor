#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "gpu_search/persistent_kernel.hh"
#include "gpu_search/pq_index.hh"

namespace {

using gpu_search::u32;
using gpu_search::u64;
using gpu_search::u8;
using gpu_search::f32;

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
    MappedRing<gpu_search::DeltaPublishDescriptor> delta_submissions(
      8, MappedRing<gpu_search::DeltaPublishDescriptor>::Direction::host_to_device);
    MappedRing<gpu_search::DeltaPublishCompletion> delta_completions(
      8, MappedRing<gpu_search::DeltaPublishCompletion>::Direction::device_to_host);

    constexpr u32 kCacheWays = 4;
    const u64 cache_keys_host[kCacheWays]{11, 22, 33, 44};
    u32 cache_states_host[kCacheWays]{2, 2, 2, 2};
    u64* cache_keys_device = nullptr;
    u32* cache_states_device = nullptr;
    u64* invalidation_key_device = nullptr;
    u64* anchor_graph_key_device = nullptr;
    u32* anchor_graph_state_device = nullptr;
    u32* anchor_graph_reader_device = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&cache_keys_device),
                          sizeof(cache_keys_host)), "cudaMalloc(cache keys)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&cache_states_device),
                          sizeof(cache_states_host)), "cudaMalloc(cache states)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&invalidation_key_device), sizeof(u64)),
               "cudaMalloc(invalidation key)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&anchor_graph_key_device), sizeof(u64)),
               "cudaMalloc(anchor route key)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&anchor_graph_state_device), sizeof(u32)),
               "cudaMalloc(anchor route state)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&anchor_graph_reader_device), sizeof(u32)),
               "cudaMalloc(anchor route reader)");
    const u64 invalidation_key_host = 22;
    check_cuda(cudaMemcpy(cache_keys_device, cache_keys_host, sizeof(cache_keys_host),
                          cudaMemcpyHostToDevice), "cudaMemcpy(cache keys)");
    check_cuda(cudaMemcpy(cache_states_device, cache_states_host, sizeof(cache_states_host),
                          cudaMemcpyHostToDevice), "cudaMemcpy(cache states)");
    check_cuda(cudaMemcpy(invalidation_key_device, &invalidation_key_host, sizeof(u64),
                          cudaMemcpyHostToDevice), "cudaMemcpy(invalidation key)");
    const u32 anchor_graph_ready = 2;
    check_cuda(cudaMemcpy(anchor_graph_key_device, &invalidation_key_host, sizeof(u64),
                          cudaMemcpyHostToDevice), "cudaMemcpy(anchor route key)");
    check_cuda(cudaMemcpy(anchor_graph_state_device, &anchor_graph_ready, sizeof(u32),
                          cudaMemcpyHostToDevice), "cudaMemcpy(anchor route state)");
    check_cuda(cudaMemset(anchor_graph_reader_device, 0, sizeof(u32)),
               "cudaMemset(anchor route reader)");

    gpu_search::DeviceDeltaRecord delta_records_host[2]{};
    delta_records_host[0].id = 7;
    delta_records_host[0].epoch = 1;
    delta_records_host[0].remote_node = 111;
    delta_records_host[0].anchor_bucket = 0;
    delta_records_host[1].id = 7;
    delta_records_host[1].epoch = 2;
    delta_records_host[1].remote_node = 222;
    delta_records_host[1].anchor_bucket = 0;
    delta_records_host[1].base_ordinal = 7;
    gpu_search::DeviceDeltaRecord* delta_records_device = nullptr;
    u32* delta_staging_slot_host = nullptr;
    u32* delta_staging_slot_device = nullptr;
    gpu_search::DeviceDeltaRecord* delta_staging_host = nullptr;
    gpu_search::DeviceDeltaRecord* delta_staging_device = nullptr;
    u8* delta_vector_staging_host = nullptr;
    u8* delta_vector_staging_device = nullptr;
    u8* delta_vectors_device = nullptr;
    u8* delta_codes_device = nullptr;
    f32* delta_encode_scratch_device = nullptr;
    f32* opq_matrix_device = nullptr;
    f32* pq_centroids_device = nullptr;
    u32* delta_next_device = nullptr;
    u32* delta_prev_device = nullptr;
    u32* delta_remote_positions_device = nullptr;
    u32* delta_bucket_heads_device = nullptr;
    u32* delta_count_device = nullptr;
    u32* override_keys_device = nullptr;
    u64* override_epochs_device = nullptr;
    u32* permanent_override_bits_device = nullptr;
    u64* remote_keys_device = nullptr;
    u32* remote_slots_device = nullptr;
    gpu_search::DeltaSupersedeUpdate* supersede_updates_device = nullptr;
    gpu_search::DeltaOverrideUpdate* override_updates_device = nullptr;
    gpu_search::DeltaDurableUpdate* durable_updates_device = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&delta_records_device),
                          sizeof(delta_records_host)), "cudaMalloc(delta records)");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&delta_staging_slot_host),
                             sizeof(u32), cudaHostAllocMapped),
               "cudaHostAlloc(delta staging slot)");
    check_cuda(cudaHostGetDevicePointer(
                 reinterpret_cast<void**>(&delta_staging_slot_device),
                 delta_staging_slot_host, 0),
               "cudaHostGetDevicePointer(delta staging slot)");
    *delta_staging_slot_host = 1;
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&delta_staging_host),
                             sizeof(gpu_search::DeviceDeltaRecord),
                             cudaHostAllocMapped), "cudaHostAlloc(delta staging)");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&delta_staging_device),
                                        delta_staging_host, 0),
               "cudaHostGetDevicePointer(delta staging)");
    *delta_staging_host = delta_records_host[1];
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&delta_vector_staging_host),
                             128, cudaHostAllocMapped),
               "cudaHostAlloc(delta vector staging)");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&delta_vector_staging_device),
                                        delta_vector_staging_host, 0),
               "cudaHostGetDevicePointer(delta vector staging)");
    for (u32 dimension = 0; dimension < 128; ++dimension) {
      delta_vector_staging_host[dimension] = static_cast<u8>((dimension * 13 + 7) % 251);
    }
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&delta_vectors_device), 2 * 128),
               "cudaMalloc(delta vectors)");
    const u32 test_subquantizers = argc > 4
      ? static_cast<u32>(std::max(1, std::atoi(argv[4]))) : 16;
    if (test_subquantizers > gpu_search::kPersistentMaxSubquantizers ||
        128 % test_subquantizers != 0) {
      throw std::invalid_argument("invalid test PQ subquantizer count");
    }
    const u32 test_subvector_dim = 128 / test_subquantizers;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&delta_codes_device),
                          2 * test_subquantizers),
               "cudaMalloc(delta codes)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&delta_encode_scratch_device),
                          128 * sizeof(f32)),
               "cudaMalloc(delta encode scratch)");
    gpu_search::pq::Model pq_model;
    pq_model.dim = 128;
    pq_model.subquantizers = test_subquantizers;
    pq_model.rotation.assign(128 * 128, 0.0f);
    for (u32 row = 0; row < 128; ++row) {
      pq_model.rotation[static_cast<size_t>(row) * 128 + (row * 17) % 128] = 1.0f;
    }
    pq_model.centroids.resize(
      static_cast<size_t>(test_subquantizers) * 256 * test_subvector_dim);
    for (u32 subquantizer = 0; subquantizer < test_subquantizers; ++subquantizer) {
      for (u32 centroid = 0; centroid < 256; ++centroid) {
        for (u32 dimension = 0; dimension < test_subvector_dim; ++dimension) {
          pq_model.centroids[
            (static_cast<size_t>(subquantizer) * 256 + centroid) *
              test_subvector_dim + dimension] =
              static_cast<f32>((centroid * 3 + subquantizer * 11 + dimension * 5) % 251);
        }
      }
    }
    std::vector<f32> decoded_vector(128);
    for (u32 dimension = 0; dimension < 128; ++dimension) {
      decoded_vector[dimension] = static_cast<f32>(delta_vector_staging_host[dimension]);
    }
    std::vector<u8> expected_delta_code(test_subquantizers);
    std::vector<f32> transformed_scratch(128);
    gpu_search::pq::encode(pq_model, decoded_vector, expected_delta_code,
                           transformed_scratch);
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&opq_matrix_device),
                          pq_model.rotation.size() * sizeof(f32)),
               "cudaMalloc(OPQ matrix)");
    check_cuda(cudaMemcpy(opq_matrix_device, pq_model.rotation.data(),
                          pq_model.rotation.size() * sizeof(f32), cudaMemcpyHostToDevice),
               "cudaMemcpy(OPQ matrix)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&pq_centroids_device),
                          pq_model.centroids.size() * sizeof(f32)),
               "cudaMalloc(PQ centroids)");
    check_cuda(cudaMemcpy(pq_centroids_device, pq_model.centroids.data(),
                          pq_model.centroids.size() * sizeof(f32), cudaMemcpyHostToDevice),
               "cudaMemcpy(PQ centroids)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&delta_next_device), 2 * sizeof(u32)),
               "cudaMalloc(delta next)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&delta_prev_device), 2 * sizeof(u32)),
               "cudaMalloc(delta prev)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&delta_remote_positions_device),
                          2 * sizeof(u32)),
               "cudaMalloc(delta remote positions)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&delta_bucket_heads_device), sizeof(u32)),
               "cudaMalloc(delta bucket heads)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&delta_count_device), sizeof(u32)),
               "cudaMalloc(delta count)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&override_keys_device), 4 * sizeof(u32)),
               "cudaMalloc(override keys)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&override_epochs_device), 4 * sizeof(u64)),
               "cudaMalloc(override epochs)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&permanent_override_bits_device),
                          sizeof(u32)),
               "cudaMalloc(permanent override bits)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&remote_keys_device), 4 * sizeof(u64)),
               "cudaMalloc(remote keys)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&remote_slots_device), 4 * sizeof(u32)),
               "cudaMalloc(remote slots)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&supersede_updates_device),
                          sizeof(gpu_search::DeltaSupersedeUpdate)),
               "cudaMalloc(supersede updates)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&override_updates_device),
                          sizeof(gpu_search::DeltaOverrideUpdate)),
               "cudaMalloc(override updates)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&durable_updates_device),
                          sizeof(gpu_search::DeltaDurableUpdate)),
               "cudaMalloc(durable updates)");
    const u32 initial_next[2]{UINT32_MAX, UINT32_MAX};
    const u32 initial_prev[2]{UINT32_MAX, UINT32_MAX};
    const u32 initial_bucket = 0;
    const u32 initial_count = 1;
    const gpu_search::DeltaSupersedeUpdate supersede_update{.slot = 0, .epoch = 2};
    const gpu_search::DeltaOverrideUpdate override_update{.ordinal = 7, .epoch = 2};
    const gpu_search::DeltaDurableUpdate durable_update{.slot = 1, .epoch = 2};
    check_cuda(cudaMemset(delta_records_device, 0, sizeof(delta_records_host)),
               "cudaMemset(delta records)");
    check_cuda(cudaMemcpy(delta_records_device, delta_records_host,
                          sizeof(gpu_search::DeviceDeltaRecord),
                          cudaMemcpyHostToDevice), "cudaMemcpy(delta record zero)");
    check_cuda(cudaMemcpy(delta_next_device, initial_next, sizeof(initial_next),
                          cudaMemcpyHostToDevice), "cudaMemcpy(delta next)");
    check_cuda(cudaMemcpy(delta_prev_device, initial_prev, sizeof(initial_prev),
                          cudaMemcpyHostToDevice), "cudaMemcpy(delta prev)");
    check_cuda(cudaMemset(delta_remote_positions_device, 0xff, 2 * sizeof(u32)),
               "cudaMemset(delta remote positions)");
    check_cuda(cudaMemcpy(delta_bucket_heads_device, &initial_bucket, sizeof(initial_bucket),
                          cudaMemcpyHostToDevice), "cudaMemcpy(delta bucket head)");
    check_cuda(cudaMemcpy(delta_count_device, &initial_count, sizeof(initial_count),
                          cudaMemcpyHostToDevice), "cudaMemcpy(delta count)");
    check_cuda(cudaMemset(override_keys_device, 0xff, 4 * sizeof(u32)),
               "cudaMemset(override keys)");
    check_cuda(cudaMemset(override_epochs_device, 0, 4 * sizeof(u64)),
               "cudaMemset(override epochs)");
    check_cuda(cudaMemset(permanent_override_bits_device, 0, sizeof(u32)),
               "cudaMemset(permanent override bits)");
    check_cuda(cudaMemset(remote_keys_device, 0, 4 * sizeof(u64)),
               "cudaMemset(remote keys)");
    check_cuda(cudaMemset(remote_slots_device, 0xff, 4 * sizeof(u32)),
               "cudaMemset(remote slots)");
    check_cuda(cudaMemcpy(supersede_updates_device, &supersede_update,
                          sizeof(supersede_update), cudaMemcpyHostToDevice),
               "cudaMemcpy(supersede update)");
    check_cuda(cudaMemcpy(override_updates_device, &override_update,
                          sizeof(override_update), cudaMemcpyHostToDevice),
               "cudaMemcpy(override update)");
    check_cuda(cudaMemcpy(durable_updates_device, &durable_update,
                          sizeof(durable_update), cudaMemcpyHostToDevice),
               "cudaMemcpy(durable update)");

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
    cudaStream_t control_stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
    check_cuda(cudaStreamCreateWithFlags(&control_stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags(control)");

    gpu_search::PersistentKernelParams params{};
    params.submissions = submissions.device_view();
    params.completions = completions.device_view();
    params.delta_submissions = delta_submissions.device_view();
    params.delta_completions = delta_completions.device_view();
    params.delta_records = delta_records_device;
    params.delta_staging_slots = delta_staging_slot_device;
    params.delta_staging_records = delta_staging_device;
    params.delta_vectors = delta_vectors_device;
    params.delta_staging_vectors = delta_vector_staging_device;
    params.delta_pq_codes = delta_codes_device;
    params.delta_encode_scratch = delta_encode_scratch_device;
    params.delta_next = delta_next_device;
    params.delta_prev = delta_prev_device;
    params.delta_remote_positions = delta_remote_positions_device;
    params.delta_bucket_heads = delta_bucket_heads_device;
    params.delta_count = delta_count_device;
    params.delta_capacity = 2;
    params.base_override_keys = override_keys_device;
    params.base_override_epochs = override_epochs_device;
    params.base_override_capacity = 4;
    params.permanent_override_bits = permanent_override_bits_device;
    params.permanent_override_words = 1;
    params.delta_remote_keys = remote_keys_device;
    params.delta_remote_slots = remote_slots_device;
    params.delta_remote_capacity = 4;
    params.anchor_count = 1;
    params.delta_supersede_updates = supersede_updates_device;
    params.delta_override_updates = override_updates_device;
    params.delta_durable_updates = durable_updates_device;
    params.graph_invalidation_keys = invalidation_key_device;
    params.anchor_graph_keys = anchor_graph_key_device;
    params.anchor_graph_states = anchor_graph_state_device;
    params.anchor_graph_readers = anchor_graph_reader_device;
    params.anchor_graph_count = 1;
    params.graph_cache_keys = cache_keys_device;
    params.graph_cache_states = cache_states_device;
    params.graph_cache_sets = 1;
    params.graph_cache_ways = kCacheWays;
    params.stop = stop_device;
    params.query_slots = 1;
    params.num_nodes = 32;
    params.dim = 128;
    params.vector_bytes = 128;
    params.vector_dtype = 1;
    params.pq_subquantizers = test_subquantizers;
    params.pq_subvector_dim = test_subvector_dim;
    params.pq_code_bytes = test_subquantizers;
    params.opq_matrix = opq_matrix_device;
    params.pq_centroids = pq_centroids_device;
    const u32 block_count = argc > 3
      ? static_cast<u32>(std::max(1, std::atoi(argv[3]))) : 8;
    const u32 query_threads = argc > 5
      ? static_cast<u32>(std::max(32, std::atoi(argv[5]))) : 128;
    if (query_threads != 128 && query_threads != 256) {
      throw std::invalid_argument("query threads must be 128 or 256");
    }
    auto query_params = params;
    query_params.delta_submissions = {};
    query_params.delta_completions = {};
    gpu_search::launch_persistent_search(stream, query_params, block_count, query_threads);
    check_cuda(cudaPeekAtLastError(), "launch_persistent_search");
    auto control_params = params;
    control_params.submissions = {};
    control_params.completions = {};
    gpu_search::launch_persistent_search(control_stream, control_params, 1, 128);
    check_cuda(cudaPeekAtLastError(), "launch_persistent_delta_control");

    const auto delta_deadline = std::chrono::steady_clock::now() +
      std::chrono::seconds(10);
    const gpu_search::DeltaPublishDescriptor delta_descriptor{
      .command_id = 99,
      .first_slot = 1,
      .record_count = 1,
      .final_count = 2,
      .invalidation_count = 1,
      .superseded_count = 1,
      .override_count = 1,
    };
    while (!delta_submissions.try_push(delta_descriptor)) {
      if (std::chrono::steady_clock::now() >= delta_deadline) {
        throw std::runtime_error("persistent delta submission ring stopped making progress");
      }
      std::this_thread::yield();
    }
    gpu_search::DeltaPublishCompletion delta_completion{};
    while (!delta_completions.try_pop(delta_completion)) {
      if (std::chrono::steady_clock::now() >= delta_deadline) {
        throw std::runtime_error("persistent delta command did not complete");
      }
      std::this_thread::yield();
    }
    if (delta_completion.command_id != 99 || delta_completion.status != 0 ||
        delta_completion.final_count != 2) {
      throw std::runtime_error("persistent delta completion is invalid");
    }

    u32 delta_count_host = 0;
    u32 delta_next_host[2]{};
    u32 delta_prev_host[2]{};
    u32 delta_bucket_head_host = UINT32_MAX;
    u32 override_keys_host[4]{};
    u64 override_epochs_host[4]{};
    u64 remote_keys_host[4]{};
    u32 remote_slots_host[4]{};
    std::vector<u8> encoded_delta_code(test_subquantizers);
    u32 anchor_graph_state_host = 0;
    check_cuda(cudaMemcpy(&delta_count_host, delta_count_device, sizeof(delta_count_host),
                          cudaMemcpyDeviceToHost), "cudaMemcpy(delta count result)");
    check_cuda(cudaMemcpy(delta_next_host, delta_next_device, sizeof(delta_next_host),
                          cudaMemcpyDeviceToHost), "cudaMemcpy(delta next result)");
    check_cuda(cudaMemcpy(delta_prev_host, delta_prev_device, sizeof(delta_prev_host),
                          cudaMemcpyDeviceToHost), "cudaMemcpy(delta prev result)");
    check_cuda(cudaMemcpy(&delta_bucket_head_host, delta_bucket_heads_device,
                          sizeof(delta_bucket_head_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(delta bucket result)");
    check_cuda(cudaMemcpy(delta_records_host, delta_records_device, sizeof(delta_records_host),
                          cudaMemcpyDeviceToHost), "cudaMemcpy(delta records result)");
    check_cuda(cudaMemcpy(cache_states_host, cache_states_device, sizeof(cache_states_host),
                          cudaMemcpyDeviceToHost), "cudaMemcpy(cache states result)");
    check_cuda(cudaMemcpy(&anchor_graph_state_host, anchor_graph_state_device,
                          sizeof(anchor_graph_state_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(anchor route state result)");
    check_cuda(cudaMemcpy(override_keys_host, override_keys_device, sizeof(override_keys_host),
                          cudaMemcpyDeviceToHost), "cudaMemcpy(override keys result)");
    check_cuda(cudaMemcpy(override_epochs_host, override_epochs_device,
                          sizeof(override_epochs_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(override epochs result)");
    check_cuda(cudaMemcpy(remote_keys_host, remote_keys_device, sizeof(remote_keys_host),
                          cudaMemcpyDeviceToHost), "cudaMemcpy(remote keys result)");
    check_cuda(cudaMemcpy(remote_slots_host, remote_slots_device, sizeof(remote_slots_host),
                          cudaMemcpyDeviceToHost), "cudaMemcpy(remote slots result)");
    check_cuda(cudaMemcpy(encoded_delta_code.data(),
                          delta_codes_device + test_subquantizers,
                          encoded_delta_code.size(), cudaMemcpyDeviceToHost),
               "cudaMemcpy(encoded delta code)");
    bool override_found = false;
    bool remote_found = false;
    for (u32 index = 0; index < 4; ++index) {
      override_found = override_found ||
        (override_keys_host[index] == 7 && override_epochs_host[index] == 2);
      remote_found = remote_found ||
        (remote_keys_host[index] == 222 && remote_slots_host[index] == 1);
    }
    if (delta_count_host != 2 || delta_bucket_head_host != 1 ||
        delta_next_host[0] != UINT32_MAX || delta_next_host[1] != UINT32_MAX ||
        delta_prev_host[0] != UINT32_MAX || delta_prev_host[1] != UINT32_MAX ||
        delta_records_host[0].superseded_epoch != 2 ||
        encoded_delta_code != expected_delta_code ||
        !override_found || !remote_found ||
        anchor_graph_state_host != 3 ||
        cache_states_host[0] != 2 || cache_states_host[1] != 3 ||
        cache_states_host[2] != 2 || cache_states_host[3] != 2) {
      throw std::runtime_error("persistent delta publication state is invalid");
    }
    const gpu_search::DeltaPublishDescriptor durable_descriptor{
      .command_id = 100,
      .first_slot = 2,
      .final_count = 2,
      .durable_count = 1,
    };
    while (!delta_submissions.try_push(durable_descriptor)) {
      if (std::chrono::steady_clock::now() >= delta_deadline) {
        throw std::runtime_error("persistent durable command queue stopped making progress");
      }
      std::this_thread::yield();
    }
    while (!delta_completions.try_pop(delta_completion)) {
      if (std::chrono::steady_clock::now() >= delta_deadline) {
        throw std::runtime_error("persistent durable command did not complete");
      }
      std::this_thread::yield();
    }
    if (delta_completion.command_id != 100 || delta_completion.status != 0 ||
        delta_completion.final_count != 2) {
      throw std::runtime_error("persistent durable completion is invalid");
    }
    check_cuda(cudaMemcpy(delta_records_host, delta_records_device,
                          sizeof(delta_records_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(marked durable records)");
    if ((delta_records_host[1].flags & gpu_search::kDeltaDurable) != 0 ||
        delta_records_host[1].superseded_epoch != delta_records_host[1].epoch ||
        (delta_records_host[0].flags & gpu_search::kDeltaDurable) != 0) {
      throw std::runtime_error("persistent durable delta retirement is invalid");
    }
    check_cuda(cudaMemcpy(&delta_bucket_head_host, delta_bucket_heads_device,
                          sizeof(delta_bucket_head_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(durable delta bucket)");
    check_cuda(cudaMemcpy(remote_keys_host, remote_keys_device,
                          sizeof(remote_keys_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(durable remote keys)");
    check_cuda(cudaMemcpy(remote_slots_host, remote_slots_device,
                          sizeof(remote_slots_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(durable remote slots)");
    check_cuda(cudaMemcpy(override_keys_host, override_keys_device,
                          sizeof(override_keys_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(durable override keys)");
    u32 durable_override_bits = 0;
    check_cuda(cudaMemcpy(&durable_override_bits, permanent_override_bits_device,
                          sizeof(durable_override_bits), cudaMemcpyDeviceToHost),
               "cudaMemcpy(durable override bits)");
    bool durable_remote_present = false;
    bool transient_override_present = false;
    for (u32 index = 0; index < 4; ++index) {
      durable_remote_present = durable_remote_present ||
        (remote_keys_host[index] == 222 && remote_slots_host[index] == 1);
      transient_override_present = transient_override_present ||
        override_keys_host[index] == 7;
    }
    if (delta_bucket_head_host != UINT32_MAX || durable_remote_present ||
        transient_override_present || (durable_override_bits & (1u << 7)) == 0) {
      throw std::runtime_error("durable delta was not retired from mutable navigation");
    }

    const auto reset_deadline = std::chrono::steady_clock::now() +
      std::chrono::seconds(10);
    const gpu_search::DeltaPublishDescriptor promote_descriptor{
      .command_id = 101,
      .first_slot = 2,
      .final_count = 2,
      .override_count = 1,
      .flags = gpu_search::kDeltaCommandPromoteOverrides,
    };
    while (!delta_submissions.try_push(promote_descriptor)) {
      if (std::chrono::steady_clock::now() >= reset_deadline) {
        throw std::runtime_error("persistent promote command queue stopped making progress");
      }
      std::this_thread::yield();
    }
    while (!delta_completions.try_pop(delta_completion)) {
      if (std::chrono::steady_clock::now() >= reset_deadline) {
        throw std::runtime_error("persistent promote command did not complete");
      }
      std::this_thread::yield();
    }
    if (delta_completion.command_id != 101 || delta_completion.status != 0 ||
        delta_completion.final_count != 2) {
      throw std::runtime_error("persistent override promotion is invalid");
    }
    const gpu_search::DeltaPublishDescriptor reset_descriptor{
      .command_id = 102,
      .record_count = 2,
      .flags = gpu_search::kDeltaCommandReset,
    };
    while (!delta_submissions.try_push(reset_descriptor)) {
      if (std::chrono::steady_clock::now() >= reset_deadline) {
        throw std::runtime_error("persistent reset command queue stopped making progress");
      }
      std::this_thread::yield();
    }
    while (!delta_completions.try_pop(delta_completion)) {
      if (std::chrono::steady_clock::now() >= reset_deadline) {
        throw std::runtime_error("persistent reset command did not complete");
      }
      std::this_thread::yield();
    }
    if (delta_completion.command_id != 102 || delta_completion.status != 0 ||
        delta_completion.final_count != 0) {
      throw std::runtime_error("persistent reset completion is invalid");
    }
    check_cuda(cudaMemcpy(&delta_count_host, delta_count_device,
                          sizeof(delta_count_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(reset delta count)");
    check_cuda(cudaMemcpy(delta_next_host, delta_next_device, sizeof(delta_next_host),
                          cudaMemcpyDeviceToHost), "cudaMemcpy(reset delta links)");
    check_cuda(cudaMemcpy(&delta_bucket_head_host, delta_bucket_heads_device,
                          sizeof(delta_bucket_head_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(reset delta bucket)");
    check_cuda(cudaMemcpy(delta_records_host, delta_records_device,
                          sizeof(delta_records_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(reset delta records)");
    check_cuda(cudaMemcpy(override_keys_host, override_keys_device,
                          sizeof(override_keys_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(reset override keys)");
    check_cuda(cudaMemcpy(override_epochs_host, override_epochs_device,
                          sizeof(override_epochs_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(reset override epochs)");
    check_cuda(cudaMemcpy(remote_keys_host, remote_keys_device,
                          sizeof(remote_keys_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(reset remote keys)");
    check_cuda(cudaMemcpy(remote_slots_host, remote_slots_device,
                          sizeof(remote_slots_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(reset remote slots)");
    for (u32 index = 0; index < 4; ++index) {
      const bool valid_override_reset =
        (override_keys_host[index] == gpu_search::kBaseOverrideEmpty ||
         override_keys_host[index] == gpu_search::kBaseOverrideTombstone) &&
        override_epochs_host[index] == 0;
      const bool valid_remote_reset =
        (remote_keys_host[index] == gpu_search::kDeltaRemoteEmpty ||
         remote_keys_host[index] == gpu_search::kDeltaRemoteTombstone) &&
        remote_slots_host[index] == UINT32_MAX;
      if (!valid_override_reset || !valid_remote_reset) {
        throw std::runtime_error("persistent reset hash state is invalid");
      }
    }
    if (delta_count_host != 0 || delta_bucket_head_host != UINT32_MAX ||
        delta_next_host[0] != UINT32_MAX || delta_next_host[1] != UINT32_MAX ||
        delta_records_host[0].remote_node != 0 ||
        delta_records_host[1].remote_node != 0) {
      throw std::runtime_error("persistent reset delta state is invalid");
    }

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
    check_cuda(cudaStreamSynchronize(control_stream), "cudaStreamSynchronize(control)");
    u32 permanent_override_bits_host = 0;
    check_cuda(cudaMemcpy(&permanent_override_bits_host,
                          permanent_override_bits_device,
                          sizeof(permanent_override_bits_host),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(permanent override bits)");
    if ((permanent_override_bits_host & (1u << 7)) == 0) {
      throw std::runtime_error("persistent override bitmap was not retained");
    }

    check_cuda(cudaFree(override_updates_device), "cudaFree(override updates)");
    check_cuda(cudaFree(durable_updates_device), "cudaFree(durable updates)");
    check_cuda(cudaFree(supersede_updates_device), "cudaFree(supersede updates)");
    check_cuda(cudaFree(remote_slots_device), "cudaFree(remote slots)");
    check_cuda(cudaFree(remote_keys_device), "cudaFree(remote keys)");
    check_cuda(cudaFree(override_epochs_device), "cudaFree(override epochs)");
    check_cuda(cudaFree(permanent_override_bits_device),
               "cudaFree(permanent override bits)");
    check_cuda(cudaFree(override_keys_device), "cudaFree(override keys)");
    check_cuda(cudaFree(delta_count_device), "cudaFree(delta count)");
    check_cuda(cudaFree(delta_bucket_heads_device), "cudaFree(delta bucket heads)");
    check_cuda(cudaFree(delta_next_device), "cudaFree(delta next)");
    check_cuda(cudaFree(delta_prev_device), "cudaFree(delta prev)");
    check_cuda(cudaFree(delta_remote_positions_device),
               "cudaFree(delta remote positions)");
    check_cuda(cudaFree(delta_records_device), "cudaFree(delta records)");
    check_cuda(cudaFreeHost(delta_staging_slot_host),
               "cudaFreeHost(delta staging slot)");
    check_cuda(cudaFree(pq_centroids_device), "cudaFree(PQ centroids)");
    check_cuda(cudaFree(opq_matrix_device), "cudaFree(OPQ matrix)");
    check_cuda(cudaFree(delta_encode_scratch_device), "cudaFree(delta encode scratch)");
    check_cuda(cudaFree(delta_codes_device), "cudaFree(delta codes)");
    check_cuda(cudaFree(delta_vectors_device), "cudaFree(delta vectors)");
    check_cuda(cudaFreeHost(delta_vector_staging_host), "cudaFreeHost(delta vector staging)");
    check_cuda(cudaFreeHost(delta_staging_host), "cudaFreeHost(delta staging)");
    check_cuda(cudaFree(invalidation_key_device), "cudaFree(invalidation key)");
    check_cuda(cudaFree(anchor_graph_reader_device), "cudaFree(anchor route reader)");
    check_cuda(cudaFree(anchor_graph_state_device), "cudaFree(anchor route state)");
    check_cuda(cudaFree(anchor_graph_key_device), "cudaFree(anchor route key)");
    check_cuda(cudaFree(cache_states_device), "cudaFree(cache states)");
    check_cuda(cudaFree(cache_keys_device), "cudaFree(cache keys)");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    check_cuda(cudaStreamDestroy(control_stream), "cudaStreamDestroy(control)");
    check_cuda(cudaFreeHost(stop_host), "cudaFreeHost(stop)");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
