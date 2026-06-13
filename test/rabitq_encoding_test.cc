#include <cuda_runtime.h>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

#include "vamana/vamana_node.hh"
#include "gpu/gpu_kernel_launcher.hh"

namespace {

float signed_dot(const float* rotated, const VamanaNode::RabitqCode& code) {
  float sum = 0.0f;
  for (u32 bit = 0; bit < VamanaNode::rabitq_code_bits(); ++bit) {
    sum += rotated[bit] * (VamanaNode::rabitq_bit(code, bit) ? 1.0f : -1.0f);
  }
  return sum;
}

bool nearly_equal(float lhs, float rhs, float tolerance) {
  return std::abs(lhs - rhs) <= tolerance * std::max(1.0f, std::max(std::abs(lhs), std::abs(rhs)));
}

bool cuda_device_available() {
  static const bool available = [] {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    return status == cudaSuccess && count > 0;
  }();
  return available;
}

bool check_layout_case(u32 dim, u32 R, VectorDType dtype, bool rabitq) {
  VamanaNode::disable_rabitq();
  VamanaNode::disable_hot_graph();
  VamanaNode::set_storage_format(vamana::StorageFormat::aos_v1);
  VamanaNode::init_static_storage(dim, R, dtype);
  if (rabitq) VamanaNode::enable_rabitq();

  const size_t vector_bytes = vector_dtype_bytes(dtype, dim);
  const size_t graph_end = VamanaNode::offset_neighbors() + VamanaNode::NEIGHBORS_SIZE;
  const size_t vector_end = VamanaNode::offset_vector() + vector_bytes;

  if (VamanaNode::storage_format_name() != "vamana_aos_v1") return false;
  if (VamanaNode::offset_neighbors() != VamanaNode::HEADER_SIZE + VamanaNode::META_SIZE) return false;
  if (VamanaNode::graph_hot_bytes() % VamanaNode::STORAGE_ALIGNMENT != 0) return false;
  if (VamanaNode::offset_vector() != VamanaNode::graph_hot_bytes()) return false;
  if (VamanaNode::offset_vector() % VamanaNode::STORAGE_ALIGNMENT != 0) return false;
  if (VamanaNode::vector_storage_bytes() % VamanaNode::STORAGE_ALIGNMENT != 0) return false;
  if (VamanaNode::total_size() % VamanaNode::STORAGE_ALIGNMENT != 0) return false;
  if (graph_end > VamanaNode::graph_hot_bytes()) return false;
  if (vector_end > (rabitq ? VamanaNode::offset_rabitq_code() : VamanaNode::total_size())) return false;
  if (VamanaNode::neighbor_read_offset() != VamanaNode::offset_id()) return false;
  if (VamanaNode::neighbor_count_offset_in_read() != VamanaNode::offset_edge_count() - VamanaNode::offset_id()) return false;
  if (VamanaNode::neighbor_payload_offset_in_read() != VamanaNode::offset_neighbors() - VamanaNode::offset_id()) return false;
  if (VamanaNode::neighbor_payload_offset_in_read() % alignof(u64) != 0) return false;
  if (VamanaNode::neighbor_read_size() != VamanaNode::offset_neighbors() + VamanaNode::NEIGHBORS_SIZE - VamanaNode::offset_id()) return false;
  if (rabitq) {
    if (VamanaNode::offset_rabitq_code() % VamanaNode::STORAGE_ALIGNMENT != 0) return false;
    if (VamanaNode::offset_rabitq_norm() % alignof(float) != 0) return false;
    if (VamanaNode::rabitq_entry_storage_size() % VamanaNode::STORAGE_ALIGNMENT != 0) return false;
  }
  return true;
}

bool check_hot_graph_case(u32 dim, u32 R, VectorDType dtype) {
  VamanaNode::disable_rabitq();
  VamanaNode::disable_hot_graph();
  VamanaNode::set_storage_format(vamana::StorageFormat::compact_v1);
  VamanaNode::init_static_storage(dim, R, dtype);
  if (VamanaNode::offset_vector() != 16 ||
      VamanaNode::total_size() % VamanaNode::COMPACT_ALIGNMENT != 0) {
    return false;
  }
  const u32 shard_bits = vamana::hot_graph::shard_bits_for(5);
  const u32 entry_size = static_cast<u32>(VamanaNode::hot_graph_entry_size());
  vec<u64> offsets{4096, 8192, 12288, 16384, 20480};
  vec<u64> counts{8, 8, 8, 8, 8};
  vec<u64> dynamic_offsets{32768, 65536, 98304, 131072, 163840};
  VamanaNode::configure_hot_graph(offsets, counts, entry_size, shard_bits, 2,
                                  dynamic_offsets,
                                  static_cast<u32>(VamanaNode::dynamic_record_size()),
                                  static_cast<u32>(VamanaNode::total_size()));
  if (!VamanaNode::HAS_HOT_GRAPH ||
      VamanaNode::storage_format_name() != "vamana_compact_v1" ||
      !VamanaNode::supports_storage_format("vamana_aos_v1") ||
      !VamanaNode::supports_storage_format("vamana_compact_v1")) {
    return false;
  }

  const RemotePtr owner{2, 16 + 3 * VamanaNode::total_size()};
  if (!VamanaNode::hot_graph_entry_available(owner) ||
      VamanaNode::hot_graph_entry_offset(owner) != offsets[2] + 3 * entry_size) {
    return false;
  }
  const RemotePtr dynamic_neighbor{4, 98765432};
  const RemotePtr base_neighbor{1, 16 + 7 * VamanaNode::total_size()};
  vec<RemotePtr> neighbors{dynamic_neighbor};
  if (R > 1) neighbors.push_back(base_neighbor);
  vec<byte_t> entry(entry_size, 0);
  VamanaNode::encode_hot_graph_entry(entry.data(), 123, static_cast<u8>(neighbors.size()),
                                     neighbors.data(), neighbors.size(), shard_bits);
  vec<byte_t> decoded(VamanaNode::neighbor_read_size(), 0);
  VamanaNode::decode_hot_graph_entry(entry.data(), decoded.data());
  const u8 edge_count = *reinterpret_cast<u8*>(
    decoded.data() + VamanaNode::neighbor_count_offset_in_read());
  const auto* slots = reinterpret_cast<const RemotePtr*>(
    decoded.data() + VamanaNode::neighbor_payload_offset_in_read());
  if (edge_count != neighbors.size() || slots[0] != dynamic_neighbor ||
      (R > 1 && slots[1] != base_neighbor)) {
    return false;
  }
  VamanaNode::disable_hot_graph();
  return true;
}

bool check_hot_graph_v2_case(u32 dim, u32 R, VectorDType dtype) {
  VamanaNode::disable_rabitq();
  VamanaNode::disable_hot_graph();
  VamanaNode::set_storage_format(vamana::StorageFormat::compact_v1);
  VamanaNode::init_static_storage(dim, R, dtype);
  const u32 shard_bits = vamana::hot_graph::shard_bits_for(4);
  const u32 entry_size = static_cast<u32>(VamanaNode::hot_graph_entry_size());
  const u32 dynamic_record_bytes = static_cast<u32>(VamanaNode::dynamic_record_size());
  const u32 dynamic_hot_offset = static_cast<u32>(VamanaNode::total_size());
  vec<u64> base_offsets{4096, 8192, 12288, 16384};
  vec<u64> base_counts{4, 4, 4, 4};
  vec<u64> dynamic_offsets{32768, 65536, 98304, 131072};
  VamanaNode::configure_hot_graph(base_offsets, base_counts, entry_size, shard_bits,
                                  2, dynamic_offsets, dynamic_record_bytes,
                                  dynamic_hot_offset);
  if (!VamanaNode::HAS_HOT_GRAPH ||
      VamanaNode::storage_format_name() != "vamana_compact_v1" ||
      VamanaNode::allocation_size() != dynamic_record_bytes) {
    return false;
  }

  const RemotePtr dynamic_owner{2, dynamic_offsets[2] + 3ull * dynamic_record_bytes};
  if (!VamanaNode::hot_graph_entry_available(dynamic_owner) ||
      VamanaNode::hot_graph_entry_offset(dynamic_owner) != dynamic_owner.byte_offset() + dynamic_hot_offset) {
    return false;
  }

  const RemotePtr neighbor{3, dynamic_offsets[3] + dynamic_record_bytes};
  vec<RemotePtr> neighbors{neighbor};
  vec<byte_t> entry(entry_size, 0);
  VamanaNode::encode_hot_graph_entry(entry.data(), 0, 1, neighbors.data(), 1,
                                     shard_bits, 17, 2);
  vec<byte_t> decoded(VamanaNode::neighbor_read_size(), 0);
  if (!VamanaNode::decode_hot_graph_entry(entry.data(), decoded.data())) {
    return false;
  }
  const auto* slots = reinterpret_cast<const RemotePtr*>(
    decoded.data() + VamanaNode::neighbor_payload_offset_in_read());
  if (slots[0] != neighbor) return false;
  entry.back() ^= 0x1u;
  if (VamanaNode::decode_hot_graph_entry(entry.data(), decoded.data())) {
    return false;
  }
  VamanaNode::disable_hot_graph();
  return true;
}

template <class T>
bool run_case(u32 dim, VectorDType dtype) {
  VamanaNode::disable_rabitq();
  VamanaNode::disable_hot_graph();
  VamanaNode::set_storage_format(vamana::StorageFormat::compact_v1);
  VamanaNode::init_static_storage(dim, 32, dtype);
  VamanaNode::enable_rabitq();
  if (VamanaNode::offset_vector() != 16 ||
      VamanaNode::size_until_vector_end() != 16 + VamanaNode::vector_bytes() ||
      VamanaNode::offset_rabitq_code() % 8 != 0 ||
      VamanaNode::total_size() % VamanaNode::COMPACT_ALIGNMENT != 0) {
    return false;
  }

  std::vector<float> centroid(dim);
  std::vector<T> vector(dim);
  std::vector<T> query(dim);
  for (u32 d = 0; d < dim; ++d) {
    centroid[d] = static_cast<float>(static_cast<int>(d % 7) - 3) * 0.25f;
    const int value = static_cast<int>(d % 31) - 15;
    if constexpr (std::is_same_v<T, float>) {
      vector[d] = static_cast<float>(value) * 0.5f;
      query[d] = static_cast<float>(value) * 0.25f + 1.0f;
    } else if constexpr (std::is_same_v<T, u8>) {
      vector[d] = static_cast<u8>(value + 64);
      query[d] = static_cast<u8>(value + 70);
    } else {
      vector[d] = static_cast<i8>(value);
      query[d] = static_cast<i8>(value + 3);
    }
  }
  const auto* query_bytes = reinterpret_cast<const byte_t*>(query.data());
  VamanaNode::set_rabitq_centroid(centroid);

  const auto* bytes = reinterpret_cast<const byte_t*>(vector.data());
  VamanaNode::RabitqCode code;
  float norm = 0.0f;
  float error = 0.0f;
  VamanaNode::compute_rabitq_entry(bytes, dtype, code, norm, error);

  std::vector<float> rotated(VamanaNode::rabitq_code_bits());
  float norm2 = 0.0f;
  VamanaNode::compute_rotated_query(query_bytes, dtype, rotated.data(), &norm2);

  if (code.size() != VamanaNode::rabitq_code_size() ||
      !nearly_equal(norm, VamanaNode::compute_rabitq_norm(bytes, dtype), 1e-5f) ||
      !(error > 0.0f) || !std::isfinite(error)) {
    return false;
  }

  const float approximate_inner_product =
      norm * signed_dot(rotated.data(), code) /
      (std::sqrt(static_cast<float>(VamanaNode::rabitq_code_bits())) * error);
  const float approximate_distance = norm2 + norm * norm - 2.0f * approximate_inner_product;
  std::vector<byte_t> entry(VamanaNode::rabitq_entry_size(), 0);
  std::memcpy(entry.data(), code.data(), code.size());
  std::memcpy(entry.data() + VamanaNode::rabitq_code_storage_size(), &norm, sizeof(norm));
  std::memcpy(entry.data() + VamanaNode::rabitq_code_storage_size() + sizeof(norm),
              &error, sizeof(error));

  if (!cuda_device_available()) {
    return true;
  }

  float* d_query = nullptr;
  byte_t* d_entry = nullptr;
  float* d_distance = nullptr;
  cudaStream_t stream{};
  cudaEvent_t event{};
  if (cudaStreamCreate(&stream) != cudaSuccess || cudaEventCreate(&event) != cudaSuccess ||
      cudaMalloc(&d_query, rotated.size() * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&d_entry, entry.size()) != cudaSuccess ||
      cudaMalloc(&d_distance, sizeof(float)) != cudaSuccess) {
    return false;
  }
  cudaMemcpyAsync(d_query, rotated.data(), rotated.size() * sizeof(float),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_entry, entry.data(), entry.size(), cudaMemcpyHostToDevice, stream);
  gpu::launch_batch_rabitq_asymmetric_distances(
      stream, event, d_query, d_entry, d_distance, norm2, 1,
      VamanaNode::rabitq_code_bits(), static_cast<u32>(code.size()),
      static_cast<u32>(entry.size()));
  float gpu_distance = 0.0f;
  cudaMemcpyAsync(&gpu_distance, d_distance, sizeof(float), cudaMemcpyDeviceToHost, stream);
  const cudaError_t gpu_status = cudaStreamSynchronize(stream);
  cudaFree(d_distance);
  cudaFree(d_entry);
  cudaFree(d_query);
  cudaEventDestroy(event);
  cudaStreamDestroy(stream);

  if (gpu_status != cudaSuccess || !std::isfinite(gpu_distance) ||
      std::abs(gpu_distance - std::max(approximate_distance, 0.0f)) >
          1e-4f * std::max(1.0f, norm2)) {
    return false;
  }
  return true;
}

}  // namespace

int main() {
  for (u32 dim : {1u, 7u, 8u, 9u, 31u, 32u, 33u, 127u, 128u,
                  129u, 300u, 511u, 512u, 513u, 1023u}) {
    for (VectorDType dtype : {VectorDType::float32, VectorDType::uint8, VectorDType::int8}) {
      for (u32 R : {1u, 16u, 32u, 63u}) {
        if (!check_layout_case(dim, R, dtype, false) ||
            !check_layout_case(dim, R, dtype, true) ||
            !check_hot_graph_case(dim, R, dtype) ||
            !check_hot_graph_v2_case(dim, R, dtype)) {
          std::cerr << "Hybrid Split layout test failed at dimension " << dim << "\n";
          return 1;
        }
      }
    }
    if (!run_case<float>(dim, VectorDType::float32) ||
        !run_case<u8>(dim, VectorDType::uint8) ||
        !run_case<i8>(dim, VectorDType::int8)) {
      std::cerr << "RaBitQ encoding test failed at dimension " << dim << "\n";
      return 1;
    }
  }
  return 0;
}
