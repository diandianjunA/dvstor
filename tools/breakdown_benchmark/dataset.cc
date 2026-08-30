#include "tools/breakdown_benchmark/dataset.hh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <cstring>
#include <stdexcept>
#include <unordered_set>

namespace tools::breakdown_benchmark {
namespace {

size_t checked_payload_bytes(size_t rows, size_t row_bytes,
                             const std::string& path) {
  if (row_bytes == 0 ||
      rows > (std::numeric_limits<size_t>::max() - 2 * sizeof(uint32_t)) /
               row_bytes) {
    throw std::runtime_error("dataset payload size overflows: " + path);
  }
  const size_t payload = rows * row_bytes;
  if (payload > static_cast<size_t>(
                  std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("dataset payload exceeds streamsize: " + path);
  }
  return payload;
}

void require_exact_file_size(const std::string& path, size_t payload_bytes,
                             const char* kind) {
  std::error_code error;
  const uintmax_t actual = std::filesystem::file_size(path, error);
  if (error) {
    throw std::runtime_error("failed to stat " + std::string(kind) +
                             " file: " + path + ": " + error.message());
  }
  const uintmax_t expected = 2 * sizeof(uint32_t) +
                             static_cast<uintmax_t>(payload_bytes);
  if (actual != expected) {
    throw std::runtime_error(
      std::string(kind) + " file size mismatch: " + path +
      " (header requires " + std::to_string(expected) +
      " bytes, file has " + std::to_string(actual) + ")");
  }
}

}  // namespace


std::vector<float> make_deterministic_vector(uint32_t seed, size_t dim) {
  if (dim == 0) {
    throw std::invalid_argument("deterministic vector dimension must be > 0");
  }
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
  if (dim != 0 && ids.size() > std::numeric_limits<size_t>::max() / dim) {
    throw std::overflow_error("deterministic dataset size overflows");
  }
  std::vector<float> vectors;
  vectors.reserve(ids.size() * dim);
  for (uint32_t id : ids) {
    auto vector = make_deterministic_vector(id, dim);
    vectors.insert(vectors.end(), vector.begin(), vector.end());
  }
  return vectors;
}

const byte_t* VectorRows::raw_row(size_t index) const {
  if (index >= count) throw std::out_of_range("vector row index out of range");
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
  const auto inferred_dtype = infer_vector_dtype_from_path(filepath_t{path});
  if (!inferred_dtype.has_value()) {
    throw std::runtime_error(
      "ambiguous or unsupported vector suffix; use .fbin, .u8bin, or "
      ".i8bin: " + path);
  }
  rows.dtype = *inferred_dtype;
  rows.dim = dim;
  rows.count = count;
  rows.vector_bytes = vector_dtype_bytes(rows.dtype, dim);
  const size_t payload_bytes =
    checked_payload_bytes(count, rows.vector_bytes, path);
  require_exact_file_size(path, payload_bytes, "vector");
  rows.raw.resize(payload_bytes);
  input.read(reinterpret_cast<char*>(rows.raw.data()),
             static_cast<std::streamsize>(rows.raw.size()));
  if (!input) {
    throw std::runtime_error("failed to read vector payload: " + path);
  }

  if (rows.dtype == VectorDType::float32) {
    for (size_t component = 0;
         component < rows.raw.size() / sizeof(float); ++component) {
      float value = 0.0f;
      std::memcpy(&value, rows.raw.data() + component * sizeof(float),
                  sizeof(value));
      if (!floating_value_is_finite(value)) {
        throw std::runtime_error(
          "float32 vector file contains a non-finite component: " + path);
      }
    }
  }

  if (decode_rows) {
    if (static_cast<size_t>(count) >
        std::numeric_limits<size_t>::max() / dim / sizeof(float)) {
      throw std::runtime_error("decoded vector size overflows: " + path);
    }
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
  if (index >= rows) {
    throw std::out_of_range("groundtruth row index out of range");
  }
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

  const size_t payload_bytes = checked_payload_bytes(
    groundtruth.rows,
    static_cast<size_t>(groundtruth.top_k) * sizeof(uint32_t), path);
  require_exact_file_size(path, payload_bytes, "groundtruth");
  groundtruth.ids.resize(payload_bytes / sizeof(uint32_t));
  input.read(reinterpret_cast<char*>(groundtruth.ids.data()),
             static_cast<std::streamsize>(payload_bytes));
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
