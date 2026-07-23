#include "tools/breakdown_benchmark/dataset.hh"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

namespace tools::breakdown_benchmark {

std::vector<float> make_deterministic_vector(uint32_t seed, size_t dim) {
  std::vector<float> vector(dim, 0.0f);
  uint64_t state = 1469598103934665603ull ^ static_cast<uint64_t>(seed);
  for (size_t index = 0; index < dim; ++index) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    const uint32_t value = static_cast<uint32_t>(
      (state * 2685821657736338717ull) >> 32);
    vector[index] = static_cast<float>(value % 10000) / 10000.0f;
  }
  vector[seed % dim] += 4.0f;
  vector[(seed * 17 + 3) % dim] += 1.0f;
  return vector;
}

std::vector<float> make_dataset(const std::vector<uint32_t>& ids, size_t dim) {
  std::vector<float> vectors;
  vectors.reserve(ids.size() * dim);
  for (uint32_t id : ids) {
    auto vector = make_deterministic_vector(id, dim);
    vectors.insert(vectors.end(), vector.begin(), vector.end());
  }
  return vectors;
}

const byte_t* VectorRows::raw_row(size_t index) const {
  return raw.data() + index * vector_bytes;
}

VectorRows read_vector_rows(const std::string& path, bool decode_rows) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open " + path);
  }

  uint32_t count = 0;
  uint32_t dim = 0;
  input.read(reinterpret_cast<char*>(&count), sizeof(count));
  input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  if (!input) {
    throw std::runtime_error("failed to read vector file header: " + path);
  }
  if (count == 0 || dim == 0) {
    throw std::runtime_error(
      "vector file must contain at least one non-empty row: " + path);
  }

  VectorRows rows;
  rows.dtype = resolve_vector_dtype_config("auto", filepath_t{path});
  rows.dim = dim;
  rows.count = count;
  rows.vector_bytes = vector_dtype_bytes(rows.dtype, dim);
  rows.raw.resize(static_cast<size_t>(count) * rows.vector_bytes);
  input.read(reinterpret_cast<char*>(rows.raw.data()),
             static_cast<std::streamsize>(rows.raw.size()));
  if (!input) {
    throw std::runtime_error("failed to read vector payload: " + path);
  }

  if (decode_rows) {
    rows.decoded.resize(static_cast<size_t>(count) * dim);
    for (size_t row = 0; row < rows.count; ++row) {
      decode_storage_vector_to_float(rows.raw_row(row), rows.dtype, rows.dim,
                                     rows.decoded.data() + row * rows.dim);
    }
  }
  return rows;
}

SinglePassRowStream::SinglePassRowStream(size_t row_count)
    : row_count_(row_count) {}

std::optional<size_t> SinglePassRowStream::try_claim() {
  if (exhausted_.load(std::memory_order_acquire)) return std::nullopt;
  const size_t row = next_row_.fetch_add(1, std::memory_order_relaxed);
  if (row >= row_count_) {
    exhausted_.store(true, std::memory_order_release);
    return std::nullopt;
  }
  return row;
}

bool SinglePassRowStream::exhausted() const {
  return exhausted_.load(std::memory_order_acquire);
}

size_t SinglePassRowStream::consumed() const {
  return std::min(next_row_.load(std::memory_order_relaxed), row_count_);
}

size_t SinglePassRowStream::capacity() const {
  return row_count_;
}

const uint32_t* GroundTruth::row(size_t index) const {
  return ids.data() + index * top_k;
}

GroundTruth read_groundtruth_bin(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open groundtruth file: " + path);
  }

  GroundTruth groundtruth;
  input.read(reinterpret_cast<char*>(&groundtruth.rows),
             sizeof(groundtruth.rows));
  input.read(reinterpret_cast<char*>(&groundtruth.top_k),
             sizeof(groundtruth.top_k));
  if (!input || groundtruth.rows == 0 || groundtruth.top_k == 0) {
    throw std::runtime_error("failed to read groundtruth header: " + path);
  }

  groundtruth.ids.resize(
    static_cast<size_t>(groundtruth.rows) * groundtruth.top_k);
  input.read(reinterpret_cast<char*>(groundtruth.ids.data()),
             static_cast<std::streamsize>(groundtruth.ids.size() * sizeof(uint32_t)));
  if (!input) {
    throw std::runtime_error("failed to read groundtruth ids: " + path);
  }
  return groundtruth;
}

double recall_at(const std::vector<uint32_t>& results,
                 const uint32_t* groundtruth,
                 uint32_t k) {
  std::unordered_set<uint32_t> truth;
  truth.reserve(k);
  for (uint32_t index = 0; index < k; ++index) truth.insert(groundtruth[index]);

  uint32_t hits = 0;
  const size_t result_count = std::min<size_t>(results.size(), k);
  for (size_t index = 0; index < result_count; ++index) {
    if (truth.contains(results[index])) ++hits;
  }
  return static_cast<double>(hits) / static_cast<double>(k);
}

}  // namespace tools::breakdown_benchmark
