#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "common/vector_dtype.hh"

namespace tools::breakdown_benchmark {

std::vector<float> make_deterministic_vector(uint32_t seed, size_t dim);
std::vector<float> make_dataset(const std::vector<uint32_t>& ids, size_t dim);

struct VectorRows {
  VectorDType dtype{VectorDType::float32};
  uint32_t dim{};
  size_t count{};
  size_t vector_bytes{};
  std::vector<byte_t> raw;
  std::vector<float> decoded;

  const byte_t* raw_row(size_t index) const;
};

VectorRows read_vector_rows(const std::string& path, bool decode_rows);

class SinglePassRowStream {
public:
  explicit SinglePassRowStream(size_t row_count);

  std::optional<size_t> try_claim();
  bool exhausted() const;
  size_t consumed() const;
  size_t capacity() const;

private:
  size_t row_count_{};
  std::atomic<size_t> next_row_{0};
  std::atomic<bool> exhausted_{false};
};

struct GroundTruth {
  uint32_t rows{};
  uint32_t top_k{};
  std::vector<uint32_t> ids;

  const uint32_t* row(size_t index) const;
};

GroundTruth read_groundtruth_bin(const std::string& path);
double recall_at(const std::vector<uint32_t>& results,
                 const uint32_t* groundtruth,
                 uint32_t k);

}  // namespace tools::breakdown_benchmark
