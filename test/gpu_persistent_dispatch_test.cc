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
#include "gpu_search/mapped_ring.hh"
#include "gpu_search/pq_index.hh"

namespace {

using gpu_search::u32;
using gpu_search::u64;
using gpu_search::u8;
using gpu_search::u16;
using gpu_search::i32;
using gpu_search::f32;
using gpu_search::MappedRing;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <class T>
class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t count = 1) : count_(count) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
               "cudaMalloc(valid query buffer)");
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) cudaFree(data_);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() const { return data_; }
  size_t bytes() const { return count_ * sizeof(T); }

private:
  T* data_{};
  size_t count_{};
};

template <class T>
class MappedValue {
public:
  MappedValue() {
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&host_), sizeof(T),
                             cudaHostAllocMapped),
               "cudaHostAlloc(valid query value)");
    *host_ = {};
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_), host_, 0),
               "cudaHostGetDevicePointer(valid query value)");
  }

  ~MappedValue() {
    if (host_ != nullptr) cudaFreeHost(host_);
  }

  T* host() const { return host_; }
  T* device() const { return device_; }

private:
  T* host_{};
  T* device_{};
};

u16 graph_checksum(const u8* data, size_t bytes) {
  u32 hash = 2166136261u;
  for (size_t index = 0; index < bytes; ++index) {
    if (index == 2 || index == 3) continue;
    hash ^= data[index];
    hash *= 16777619u;
  }
  hash ^= hash >> 16;
  return static_cast<u16>(hash);
}

void run_valid_resident_query_test(u32 subquantizers, u32 query_threads) {
  constexpr u32 dim = 128;
  constexpr u32 graph_entry_bytes = 16;
  constexpr u32 node_record_bytes = 16 + dim;
  constexpr u32 visited_capacity = 256;
  constexpr u32 request_capacity = gpu_search::kPersistentMaxMergeCandidates;

  MappedRing<gpu_search::QueryDescriptor> submissions(
    8, MappedRing<gpu_search::QueryDescriptor>::Direction::host_to_device);
  MappedRing<gpu_search::CompletionDescriptor> completions(
    8, MappedRing<gpu_search::CompletionDescriptor>::Direction::device_to_host);
  MappedValue<u32> stop;
  MappedValue<u32> direct_disabled;
  MappedValue<i32> direct_error;

  DeviceBuffer<gpu_search::DeviceShardRegion> shards;
  DeviceBuffer<u8> pq_codes(subquantizers);
  DeviceBuffer<f32> centroids(static_cast<size_t>(subquantizers) * 256 *
                              (dim / subquantizers));
  DeviceBuffer<u32> entry_points;
  DeviceBuffer<u64> route_keys;
  DeviceBuffer<u8> route_records(graph_entry_bytes);
  DeviceBuffer<u32> route_states;
  DeviceBuffer<u32> route_readers;
  DeviceBuffer<u8> query(dim);
  DeviceBuffer<u32> result_ids;
  DeviceBuffer<f32> result_distances;
  DeviceBuffer<f32> decoded(dim);
  DeviceBuffer<f32> transformed(dim);
  DeviceBuffer<f32> query_lut(static_cast<size_t>(subquantizers) * 256);
  DeviceBuffer<u32> navigation_handles(request_capacity);
  DeviceBuffer<f32> navigation_distances(request_capacity);
  DeviceBuffer<u32> visited(visited_capacity);
  DeviceBuffer<u8> dynamic_codes(static_cast<size_t>(request_capacity) * subquantizers);
  DeviceBuffer<u32> request_shards(request_capacity);
  DeviceBuffer<u64> request_offsets(request_capacity);
  DeviceBuffer<u64> request_iovas(request_capacity);
  DeviceBuffer<u8> exact_records(node_record_bytes);
  DeviceBuffer<u8> exact_cache(node_record_bytes);
  DeviceBuffer<u32> exact_cache_keys;
  DeviceBuffer<u32> exact_cache_states;
  DeviceBuffer<u32> exact_cache_readers;
  DeviceBuffer<u32> exact_cache_victims;
  DeviceBuffer<u32> delta_count;

  const gpu_search::DeviceShardRegion shard{
    .ordinal_base = 0,
    .node_count = 1,
    .node_base_offset = 64,
    .node_stride = node_record_bytes,
    .graph_base_offset = 512,
    .memory_node = 0,
  };
  const u32 entry_point = 0;
  const u64 route_key = 512;
  const u32 ready = 2;
  const u32 zero = 0;
  std::vector<u8> route_record(graph_entry_bytes, 0);
  const u16 checksum = graph_checksum(route_record.data(), route_record.size());
  route_record[2] = static_cast<u8>(checksum);
  route_record[3] = static_cast<u8>(checksum >> 8);
  std::vector<u8> exact_record(node_record_bytes, 0);
  const u32 expected_id = 42;
  std::memcpy(exact_record.data() + 8, &expected_id, sizeof(expected_id));

  check_cuda(cudaMemcpy(shards.get(), &shard, sizeof(shard), cudaMemcpyHostToDevice),
             "cudaMemcpy(valid query shard)");
  check_cuda(cudaMemset(pq_codes.get(), 0, pq_codes.bytes()),
             "cudaMemset(valid query codes)");
  check_cuda(cudaMemset(centroids.get(), 0, centroids.bytes()),
             "cudaMemset(valid query centroids)");
  check_cuda(cudaMemcpy(entry_points.get(), &entry_point, sizeof(entry_point),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(valid query entry point)");
  check_cuda(cudaMemcpy(route_keys.get(), &route_key, sizeof(route_key),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(valid query route key)");
  check_cuda(cudaMemcpy(route_records.get(), route_record.data(), route_record.size(),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(valid query route record)");
  check_cuda(cudaMemcpy(route_states.get(), &ready, sizeof(ready), cudaMemcpyHostToDevice),
             "cudaMemcpy(valid query route state)");
  check_cuda(cudaMemset(route_readers.get(), 0, route_readers.bytes()),
             "cudaMemset(valid query route readers)");
  check_cuda(cudaMemset(query.get(), 0, query.bytes()), "cudaMemset(valid query)");
  check_cuda(cudaMemcpy(exact_cache.get(), exact_record.data(), exact_record.size(),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(valid query exact cache)");
  check_cuda(cudaMemcpy(exact_cache_keys.get(), &entry_point, sizeof(entry_point),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(valid query exact key)");
  check_cuda(cudaMemcpy(exact_cache_states.get(), &ready, sizeof(ready),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(valid query exact state)");
  check_cuda(cudaMemset(exact_cache_readers.get(), 0, exact_cache_readers.bytes()),
             "cudaMemset(valid query exact readers)");
  check_cuda(cudaMemset(exact_cache_victims.get(), 0, exact_cache_victims.bytes()),
             "cudaMemset(valid query exact victims)");
  check_cuda(cudaMemcpy(delta_count.get(), &zero, sizeof(zero), cudaMemcpyHostToDevice),
             "cudaMemcpy(valid query delta count)");

  gpu_search::PersistentKernelParams params{
    .submissions = submissions.device_view(),
    .device_submissions = {},
    .completions = completions.device_view(),
    .delta_submissions = {},
    .delta_completions = {},
    .shards = shards.get(),
    .num_shards = 1,
    .pq_codes = pq_codes.get(),
    .pq_centroids = centroids.get(),
    .entry_points = entry_points.get(),
    .entry_point_count = 1,
    .num_nodes = 1,
    .dim = dim,
    .pq_subquantizers = subquantizers,
    .pq_subvector_dim = dim / subquantizers,
    .pq_code_bytes = subquantizers,
    .graph_entry_bytes = graph_entry_bytes,
    .graph_degree = 1,
    .node_record_bytes = node_record_bytes,
    .vector_bytes = dim,
    .vector_dtype = 1,
    .traversal_beam_width = 1,
    .final_rerank_width = 1,
    .entry_seed_count = 1,
    .exact_width = 1,
    .max_expansions = 1,
    .prefetch_depth = 1,
    .visited_capacity = visited_capacity,
    .query_slots = 1,
    .direct_region_count = 1,
    .direct_disabled = direct_disabled.device(),
    .direct_error = direct_error.device(),
    .delta_count = delta_count.get(),
    .anchor_graph_keys = route_keys.get(),
    .anchor_graph_records = route_records.get(),
    .anchor_graph_states = route_states.get(),
    .anchor_graph_readers = route_readers.get(),
    .anchor_graph_count = 1,
    .stop = stop.device(),
    .decoded_queries = decoded.get(),
    .transformed_queries = transformed.get(),
    .query_luts = query_lut.get(),
    .navigation_candidate_handles = navigation_handles.get(),
    .navigation_candidate_distances = navigation_distances.get(),
    .visited_hash = visited.get(),
    .exact_records = exact_records.get(),
    .dynamic_code_records = dynamic_codes.get(),
    .dynamic_code_request_shards = request_shards.get(),
    .dynamic_code_request_offsets = request_offsets.get(),
    .dynamic_code_request_local_iovas = request_iovas.get(),
    .exact_cache = exact_cache.get(),
    .exact_cache_stride = node_record_bytes,
    .exact_cache_sets = 1,
    .exact_cache_ways = 1,
    .exact_cache_keys = exact_cache_keys.get(),
    .exact_cache_states = exact_cache_states.get(),
    .exact_cache_readers = exact_cache_readers.get(),
    .exact_cache_victims = exact_cache_victims.get(),
    .result_ids = result_ids.get(),
    .result_distances = result_distances.get(),
  };

  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
             "cudaStreamCreate(valid query)");
  gpu_search::launch_persistent_search(stream, params, 1, query_threads);
  check_cuda(cudaPeekAtLastError(), "launch_persistent_search(valid query)");

  const gpu_search::QueryDescriptor descriptor{
    .request_id = 1,
    .query_device_address = reinterpret_cast<u64>(query.get()),
    .result_device_address = reinterpret_cast<u64>(result_ids.get()),
    .query_slot = 0,
    .result_capacity = 1,
    .dim = dim,
    .k = 1,
    .query_dtype = 1,
  };
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!submissions.try_push(descriptor)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      std::cerr << "valid resident query submission stalled\n";
      std::_Exit(EXIT_FAILURE);
    }
    std::this_thread::yield();
  }
  gpu_search::CompletionDescriptor completion{};
  while (!completions.try_pop(completion)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      std::cerr << "valid resident query did not complete\n";
      std::_Exit(EXIT_FAILURE);
    }
    std::this_thread::yield();
  }
  u32 result_id = 0;
  check_cuda(cudaMemcpy(&result_id, result_ids.get(), sizeof(result_id),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(valid query result)");
  if (completion.status != 0 || completion.result_count != 1 ||
      completion.route_hits != 1 || result_id != expected_id) {
    throw std::runtime_error("valid resident query produced an invalid result");
  }
  std::atomic_ref<u32>(*stop.host()).store(1, std::memory_order_release);
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(valid query)");
  check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy(valid query)");
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
    delta_records_host[1].resident_pq_slot = 0;
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
    u8* resident_pq_codes_device = nullptr;
    u64* resident_pq_keys_device = nullptr;
    u32* resident_pq_slots_device = nullptr;
    u32* resident_pq_positions_device = nullptr;
    gpu_search::DeltaSupersedeUpdate* supersede_updates_device = nullptr;
    gpu_search::DeltaOverrideUpdate* override_updates_device = nullptr;
    gpu_search::DeltaDurableUpdate* durable_updates_device = nullptr;
    gpu_search::ResidentPqEraseUpdate* resident_pq_erase_updates_device = nullptr;
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
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&resident_pq_codes_device),
                          2 * test_subquantizers),
               "cudaMalloc(resident PQ codes)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&resident_pq_keys_device),
                          4 * sizeof(u64)),
               "cudaMalloc(resident PQ keys)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&resident_pq_slots_device),
                          4 * sizeof(u32)),
               "cudaMalloc(resident PQ slots)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&resident_pq_positions_device),
                          2 * sizeof(u32)),
               "cudaMalloc(resident PQ positions)");
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
    check_cuda(cudaMalloc(
                 reinterpret_cast<void**>(&resident_pq_erase_updates_device),
                 sizeof(gpu_search::ResidentPqEraseUpdate)),
               "cudaMalloc(resident PQ erase updates)");
    const u32 initial_next[2]{UINT32_MAX, UINT32_MAX};
    const u32 initial_prev[2]{UINT32_MAX, UINT32_MAX};
    const u32 initial_bucket = 0;
    const u32 initial_count = 1;
    const gpu_search::DeltaSupersedeUpdate supersede_update{.slot = 0, .epoch = 2};
    const gpu_search::DeltaOverrideUpdate override_update{.ordinal = 7, .epoch = 2};
    const gpu_search::DeltaDurableUpdate durable_update{.slot = 1, .epoch = 2};
    const gpu_search::ResidentPqEraseUpdate resident_pq_erase_update{
      .remote_node = 222,
      .slot = 0,
    };
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
    check_cuda(cudaMemset(resident_pq_codes_device, 0,
                          2 * test_subquantizers),
               "cudaMemset(resident PQ codes)");
    check_cuda(cudaMemset(resident_pq_keys_device, 0, 4 * sizeof(u64)),
               "cudaMemset(resident PQ keys)");
    check_cuda(cudaMemset(resident_pq_slots_device, 0xff, 4 * sizeof(u32)),
               "cudaMemset(resident PQ slots)");
    check_cuda(cudaMemset(resident_pq_positions_device, 0xff, 2 * sizeof(u32)),
               "cudaMemset(resident PQ positions)");
    check_cuda(cudaMemcpy(supersede_updates_device, &supersede_update,
                          sizeof(supersede_update), cudaMemcpyHostToDevice),
               "cudaMemcpy(supersede update)");
    check_cuda(cudaMemcpy(override_updates_device, &override_update,
                          sizeof(override_update), cudaMemcpyHostToDevice),
               "cudaMemcpy(override update)");
    check_cuda(cudaMemcpy(durable_updates_device, &durable_update,
                          sizeof(durable_update), cudaMemcpyHostToDevice),
               "cudaMemcpy(durable update)");
    check_cuda(cudaMemcpy(resident_pq_erase_updates_device,
                          &resident_pq_erase_update,
                          sizeof(resident_pq_erase_update),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(resident PQ erase update)");

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
    params.resident_pq_codes = resident_pq_codes_device;
    params.resident_pq_keys = resident_pq_keys_device;
    params.resident_pq_slots = resident_pq_slots_device;
    params.resident_pq_positions = resident_pq_positions_device;
    params.resident_pq_capacity = 2;
    params.resident_pq_table_capacity = 4;
    params.anchor_count = 1;
    params.delta_supersede_updates = supersede_updates_device;
    params.delta_override_updates = override_updates_device;
    params.delta_durable_updates = durable_updates_device;
    params.resident_pq_erase_updates = resident_pq_erase_updates_device;
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
    run_valid_resident_query_test(test_subquantizers, query_threads);
    auto query_params = params;
    query_params.delta_submissions = {};
    query_params.delta_completions = {};
    auto control_params = params;
    control_params.submissions = {};
    control_params.completions = {};
    gpu_search::launch_persistent_search(control_stream, control_params, 1, 128);
    check_cuda(cudaPeekAtLastError(), "launch_persistent_delta_control");
    gpu_search::launch_persistent_search(stream, query_params, block_count, query_threads);
    check_cuda(cudaPeekAtLastError(), "launch_persistent_search");

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
    u64 resident_pq_keys_host[4]{};
    u32 resident_pq_slots_host[4]{};
    u32 resident_pq_positions_host[2]{};
    std::vector<u8> encoded_delta_code(test_subquantizers);
    std::vector<u8> encoded_resident_pq_code(test_subquantizers);
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
    check_cuda(cudaMemcpy(resident_pq_keys_host, resident_pq_keys_device,
                          sizeof(resident_pq_keys_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(resident PQ keys result)");
    check_cuda(cudaMemcpy(resident_pq_slots_host, resident_pq_slots_device,
                          sizeof(resident_pq_slots_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(resident PQ slots result)");
    check_cuda(cudaMemcpy(resident_pq_positions_host,
                          resident_pq_positions_device,
                          sizeof(resident_pq_positions_host),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(resident PQ positions result)");
    check_cuda(cudaMemcpy(encoded_delta_code.data(),
                          delta_codes_device + test_subquantizers,
                          encoded_delta_code.size(), cudaMemcpyDeviceToHost),
               "cudaMemcpy(encoded delta code)");
    check_cuda(cudaMemcpy(encoded_resident_pq_code.data(),
                          resident_pq_codes_device,
                          encoded_resident_pq_code.size(), cudaMemcpyDeviceToHost),
               "cudaMemcpy(encoded resident PQ code)");
    bool override_found = false;
    bool remote_found = false;
    bool resident_pq_found = false;
    for (u32 index = 0; index < 4; ++index) {
      override_found = override_found ||
        (override_keys_host[index] == 7 && override_epochs_host[index] == 2);
      remote_found = remote_found ||
        (remote_keys_host[index] == 222 && remote_slots_host[index] == 1);
      resident_pq_found = resident_pq_found ||
        (resident_pq_keys_host[index] == 222 &&
         resident_pq_slots_host[index] == 0);
    }
    if (delta_count_host != 2 || delta_bucket_head_host != 1 ||
        delta_next_host[0] != UINT32_MAX || delta_next_host[1] != UINT32_MAX ||
        delta_prev_host[0] != UINT32_MAX || delta_prev_host[1] != UINT32_MAX ||
        delta_records_host[0].superseded_epoch != 2 ||
        encoded_delta_code != expected_delta_code ||
        encoded_resident_pq_code != expected_delta_code ||
        !override_found || !remote_found || !resident_pq_found ||
        resident_pq_positions_host[0] >= 4 ||
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
    check_cuda(cudaMemcpy(resident_pq_keys_host, resident_pq_keys_device,
                          sizeof(resident_pq_keys_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(durable resident PQ keys)");
    check_cuda(cudaMemcpy(resident_pq_slots_host, resident_pq_slots_device,
                          sizeof(resident_pq_slots_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(durable resident PQ slots)");
    u32 durable_override_bits = 0;
    check_cuda(cudaMemcpy(&durable_override_bits, permanent_override_bits_device,
                          sizeof(durable_override_bits), cudaMemcpyDeviceToHost),
               "cudaMemcpy(durable override bits)");
    bool durable_remote_present = false;
    bool transient_override_present = false;
    bool durable_resident_pq_present = false;
    for (u32 index = 0; index < 4; ++index) {
      durable_remote_present = durable_remote_present ||
        (remote_keys_host[index] == 222 && remote_slots_host[index] == 1);
      transient_override_present = transient_override_present ||
        override_keys_host[index] == 7;
      durable_resident_pq_present = durable_resident_pq_present ||
        (resident_pq_keys_host[index] == 222 &&
         resident_pq_slots_host[index] == 0);
    }
    if (delta_bucket_head_host != UINT32_MAX || durable_remote_present ||
        transient_override_present || !durable_resident_pq_present ||
        (durable_override_bits & (1u << 7)) == 0) {
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
    check_cuda(cudaMemcpy(resident_pq_keys_host, resident_pq_keys_device,
                          sizeof(resident_pq_keys_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(reset resident PQ keys)");
    check_cuda(cudaMemcpy(resident_pq_slots_host, resident_pq_slots_device,
                          sizeof(resident_pq_slots_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(reset resident PQ slots)");
    bool reset_retained_resident_pq = false;
    for (u32 index = 0; index < 4; ++index) {
      reset_retained_resident_pq = reset_retained_resident_pq ||
        (resident_pq_keys_host[index] == 222 &&
         resident_pq_slots_host[index] == 0);
    }
    if (!reset_retained_resident_pq) {
      throw std::runtime_error("L0 reset incorrectly discarded durable resident PQ");
    }

    const gpu_search::DeltaPublishDescriptor resident_pq_erase_descriptor{
      .command_id = 103,
      .resident_pq_erase_count = 1,
    };
    while (!delta_submissions.try_push(resident_pq_erase_descriptor)) {
      if (std::chrono::steady_clock::now() >= reset_deadline) {
        throw std::runtime_error("resident PQ erase command queue stopped making progress");
      }
      std::this_thread::yield();
    }
    while (!delta_completions.try_pop(delta_completion)) {
      if (std::chrono::steady_clock::now() >= reset_deadline) {
        throw std::runtime_error("resident PQ erase command did not complete");
      }
      std::this_thread::yield();
    }
    if (delta_completion.command_id != 103 || delta_completion.status != 0 ||
        delta_completion.final_count != 0) {
      throw std::runtime_error("resident PQ erase completion is invalid");
    }
    check_cuda(cudaMemcpy(resident_pq_keys_host, resident_pq_keys_device,
                          sizeof(resident_pq_keys_host), cudaMemcpyDeviceToHost),
               "cudaMemcpy(erased resident PQ keys)");
    check_cuda(cudaMemcpy(resident_pq_positions_host,
                          resident_pq_positions_device,
                          sizeof(resident_pq_positions_host),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(erased resident PQ positions)");
    for (u64 key : resident_pq_keys_host) {
      if (key == 222) {
        throw std::runtime_error("resident PQ erase left a live remote mapping");
      }
    }
    if (resident_pq_positions_host[0] != UINT32_MAX) {
      throw std::runtime_error("resident PQ erase did not recycle its slot");
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

    check_cuda(cudaFree(resident_pq_erase_updates_device),
               "cudaFree(resident PQ erase updates)");
    check_cuda(cudaFree(override_updates_device), "cudaFree(override updates)");
    check_cuda(cudaFree(durable_updates_device), "cudaFree(durable updates)");
    check_cuda(cudaFree(supersede_updates_device), "cudaFree(supersede updates)");
    check_cuda(cudaFree(remote_slots_device), "cudaFree(remote slots)");
    check_cuda(cudaFree(remote_keys_device), "cudaFree(remote keys)");
    check_cuda(cudaFree(resident_pq_positions_device),
               "cudaFree(resident PQ positions)");
    check_cuda(cudaFree(resident_pq_slots_device), "cudaFree(resident PQ slots)");
    check_cuda(cudaFree(resident_pq_keys_device), "cudaFree(resident PQ keys)");
    check_cuda(cudaFree(resident_pq_codes_device), "cudaFree(resident PQ codes)");
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
