#pragma once

#include "tools/vamana_offline/config.hh"

namespace tools::vamana_offline {

struct Dataset {
  filepath_t source_file{};
  u32 dim{0};
  size_t total_vectors{0};
  VectorDType dtype{VectorDType::float32};
  size_t vector_bytes{0};
  vec<byte_t> raw_vectors;

  size_t size() const { return vector_count; }
  const byte_t* raw_vector(size_t i) const { return raw_vectors.data() + i * vector_bytes; }
  u32 id(size_t i) const { return static_cast<u32>(i); }

private:
  size_t vector_count{0};
  friend Dataset read_dataset(const VamanaBuildConfig& config);
};

filepath_t resolve_dataset_file(const filepath_t& input_path);
Dataset read_dataset(const VamanaBuildConfig& config);
float dataset_l2_distance(const Dataset& dataset, size_t lhs, size_t rhs);
float dataset_distance(const Dataset& dataset, size_t lhs, size_t rhs);
float dataset_distance_float_query(const Dataset& dataset, const float* query, size_t rhs);
void dataset_decode_vector(const Dataset& dataset, size_t row, float* dst);

}  // namespace tools::vamana_offline
