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

bool check_layout_case(u32 dim, u32 R, VectorDType dtype, bool rabitq) {
  VamanaNode::disable_rabitq();
  VamanaNode::init_static_storage(dim, R, dtype);
  if (rabitq) VamanaNode::enable_rabitq();

  const size_t vector_bytes = vector_dtype_bytes(dtype, dim);
  const size_t graph_end = VamanaNode::offset_neighbors() + VamanaNode::NEIGHBORS_SIZE;
  const size_t vector_end = VamanaNode::offset_vector() + vector_bytes;

  if (VamanaNode::storage_format_name() != "hybrid_split_v1") return false;
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

template <class T>
bool run_case(u32 dim, VectorDType dtype) {
  VamanaNode::disable_rabitq();
  VamanaNode::init_static_storage(dim, 32, dtype);
  VamanaNode::enable_rabitq();

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
            !check_layout_case(dim, R, dtype, true)) {
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
