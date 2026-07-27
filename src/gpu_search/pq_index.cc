#include "gpu_search/pq_index.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "common/vector_dtype.hh"

namespace gpu_search::pq {
namespace {

constexpr u64 kChecksumOffset = 1469598103934665603ULL;
constexpr u64 kChecksumPrime = 1099511628211ULL;

bool fail(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
  return false;
}

u64 checksum_update(u64 state, const void* data, size_t bytes) {
  const auto* source = static_cast<const u8*>(data);
  for (size_t index = 0; index < bytes; ++index) {
    state ^= source[index];
    state *= kChecksumPrime;
  }
  return state;
}

bool read_exact(std::istream& input, void* destination, size_t bytes) {
  input.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
  return static_cast<size_t>(input.gcount()) == bytes;
}

f32 squared_l2_saturated(const f32* lhs, const f32* rhs, u32 dim) {
  f32 distance = 0.0f;
  for (u32 dimension = 0; dimension < dim; ++dimension) {
    const f32 difference = lhs[dimension] - rhs[dimension];
    distance += difference * difference;
  }
  if (floating_value_is_finite(distance) &&
      distance < std::numeric_limits<f32>::max()) {
    return std::min(distance, kMaxValidSquaredL2);
  }
  f64 wide_distance = 0.0;
  for (u32 dimension = 0; dimension < dim; ++dimension) {
    const f64 difference = static_cast<f64>(lhs[dimension]) -
      static_cast<f64>(rhs[dimension]);
    wide_distance += difference * difference;
  }
  return saturate_squared_l2(wide_distance);
}

}  // namespace

u64 Model::checksum() const {
  u64 state = kChecksumOffset;
  if (!rotation.empty()) {
    state = checksum_update(state, rotation.data(), rotation.size() * sizeof(f32));
  }
  if (!centroids.empty()) {
    state = checksum_update(state, centroids.data(), centroids.size() * sizeof(f32));
  }
  return state;
}

bool validate(const Model& model, std::string* error) {
  if (model.dim == 0 || model.subquantizers == 0 ||
      model.dim % model.subquantizers != 0) {
    return fail(error, "PQ model dimension must be divisible by its subquantizer count");
  }
  if (model.bits_per_code != kBitsPerCode) {
    return fail(error, "PQ runtime supports exactly 8 bits per subquantizer");
  }
  const size_t expected_rotation = static_cast<size_t>(model.dim) * model.dim;
  if (!model.rotation.empty() && model.rotation.size() != expected_rotation) {
    return fail(error, "PQ model rotation matrix has an invalid shape");
  }
  const size_t expected_centroids = static_cast<size_t>(model.subquantizers) *
    kCentroidsPerSubquantizer * model.subvector_dim();
  if (model.centroids.size() != expected_centroids) {
    return fail(error, "PQ model centroid table has an invalid shape");
  }
  const auto finite = [](f32 value) {
    return floating_value_is_finite(value);
  };
  if (!std::all_of(model.rotation.begin(), model.rotation.end(), finite) ||
      !std::all_of(model.centroids.begin(), model.centroids.end(), finite)) {
    return fail(error, "PQ model contains non-finite values");
  }
  return true;
}

bool write_model(const std::filesystem::path& path, const Model& model,
                 std::string* error) {
  if (!validate(model, error)) return false;
  ModelHeader header;
  header.dim = model.dim;
  header.subquantizers = model.subquantizers;
  header.bits_per_code = model.bits_per_code;
  header.subvector_dim = model.subvector_dim();
  header.code_bytes = model.code_bytes();
  header.flags = model.has_rotation() ? kFlagHasRotation : 0;
  header.rotation_offset = sizeof(ModelHeader);
  header.rotation_bytes = model.rotation.size() * sizeof(f32);
  header.centroids_offset = header.rotation_offset + header.rotation_bytes;
  header.centroids_bytes = model.centroids.size() * sizeof(f32);
  header.file_bytes = header.centroids_offset + header.centroids_bytes;
  header.payload_checksum = model.checksum();

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.good()) return fail(error, "failed to create PQ model: " + path.string());
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  if (!model.rotation.empty()) {
    output.write(reinterpret_cast<const char*>(model.rotation.data()),
                 static_cast<std::streamsize>(header.rotation_bytes));
  }
  output.write(reinterpret_cast<const char*>(model.centroids.data()),
               static_cast<std::streamsize>(header.centroids_bytes));
  if (!output.good()) return fail(error, "failed to write PQ model: " + path.string());
  return true;
}

bool read_model(const std::filesystem::path& path, Model& model,
                std::string* error) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) return fail(error, "missing PQ model: " + path.string());
  ModelHeader header;
  if (!read_exact(input, &header, sizeof(header)) || header.magic != kModelMagic ||
      header.version != kModelVersion || header.header_bytes != sizeof(ModelHeader) ||
      header.endian_marker != kEndianMarker || header.bits_per_code != kBitsPerCode ||
      header.code_bytes != header.subquantizers || header.subvector_dim == 0 ||
      header.dim != header.subquantizers * header.subvector_dim ||
      header.rotation_offset != sizeof(ModelHeader) ||
      header.centroids_offset != header.rotation_offset + header.rotation_bytes ||
      header.file_bytes != header.centroids_offset + header.centroids_bytes) {
    return fail(error, "invalid PQ model header: " + path.string());
  }
  const bool has_rotation = (header.flags & kFlagHasRotation) != 0;
  const u64 expected_rotation_bytes = has_rotation
    ? static_cast<u64>(header.dim) * header.dim * sizeof(f32) : 0;
  const u64 expected_centroid_bytes = static_cast<u64>(header.subquantizers) *
    kCentroidsPerSubquantizer * header.subvector_dim * sizeof(f32);
  if (header.rotation_bytes != expected_rotation_bytes ||
      header.centroids_bytes != expected_centroid_bytes) {
    return fail(error, "invalid PQ model payload shape: " + path.string());
  }
  input.seekg(0, std::ios::end);
  if (static_cast<u64>(input.tellg()) != header.file_bytes) {
    return fail(error, "truncated PQ model: " + path.string());
  }
  input.seekg(static_cast<std::streamoff>(header.rotation_offset));
  Model loaded;
  loaded.dim = header.dim;
  loaded.subquantizers = header.subquantizers;
  loaded.bits_per_code = header.bits_per_code;
  loaded.rotation.resize(static_cast<size_t>(header.rotation_bytes / sizeof(f32)));
  loaded.centroids.resize(static_cast<size_t>(header.centroids_bytes / sizeof(f32)));
  if ((!loaded.rotation.empty() &&
       !read_exact(input, loaded.rotation.data(), header.rotation_bytes)) ||
      !read_exact(input, loaded.centroids.data(), header.centroids_bytes)) {
    return fail(error, "failed to read PQ model payload: " + path.string());
  }
  if (loaded.checksum() != header.payload_checksum) {
    return fail(error, "PQ model checksum mismatch: " + path.string());
  }
  if (!validate(loaded, error)) return false;
  model = std::move(loaded);
  return true;
}

void transform(const Model& model, std::span<const f32> input,
               std::span<f32> output) {
  if (input.size() != model.dim || output.size() != model.dim) {
    throw std::invalid_argument("PQ transform dimension mismatch");
  }
  if (!std::all_of(input.begin(), input.end(), [](f32 value) {
        return floating_value_is_finite(value);
      })) {
    throw std::invalid_argument("PQ transform input must be finite");
  }
  if (!model.has_rotation()) {
    std::copy(input.begin(), input.end(), output.begin());
    return;
  }
  for (u32 row = 0; row < model.dim; ++row) {
    f32 value = 0.0f;
    const f32* matrix_row = model.rotation.data() + static_cast<size_t>(row) * model.dim;
    for (u32 column = 0; column < model.dim; ++column) {
      value += matrix_row[column] * input[column];
    }
    if (!floating_value_is_finite(value)) {
      f64 wide_value = 0.0;
      for (u32 column = 0; column < model.dim; ++column) {
        wide_value = std::fma(
          static_cast<f64>(matrix_row[column]),
          static_cast<f64>(input[column]), wide_value);
      }
      const f64 maximum = static_cast<f64>(
        std::numeric_limits<f32>::max());
      value = wide_value >= maximum
        ? std::numeric_limits<f32>::max()
        : wide_value <= -maximum
          ? -std::numeric_limits<f32>::max()
          : static_cast<f32>(wide_value);
    }
    output[row] = value;
  }
}

void encode(const Model& model, std::span<const f32> input,
            std::span<u8> code, std::span<f32> transformed_scratch) {
  if (code.size() != model.code_bytes() || transformed_scratch.size() != model.dim) {
    throw std::invalid_argument("PQ encode buffer shape mismatch");
  }
  transform(model, input, transformed_scratch);
  const u32 dsub = model.subvector_dim();
  for (u32 subquantizer = 0; subquantizer < model.subquantizers; ++subquantizer) {
    const f32* value = transformed_scratch.data() + static_cast<size_t>(subquantizer) * dsub;
    const f32* table = model.centroids.data() +
      static_cast<size_t>(subquantizer) * kCentroidsPerSubquantizer * dsub;
    f32 best_distance = std::numeric_limits<f32>::max();
    u32 best = 0;
    for (u32 centroid = 0; centroid < kCentroidsPerSubquantizer; ++centroid) {
      const f32* candidate = table + static_cast<size_t>(centroid) * dsub;
      const f32 distance = squared_l2_saturated(value, candidate, dsub);
      if (distance < best_distance) {
        best_distance = distance;
        best = centroid;
      }
    }
    code[subquantizer] = static_cast<u8>(best);
  }
}

void build_distance_table(const Model& model, std::span<const f32> input,
                          std::span<f32> table,
                          std::span<f32> transformed_scratch) {
  const size_t table_size = static_cast<size_t>(model.subquantizers) *
    kCentroidsPerSubquantizer;
  if (table.size() != table_size || transformed_scratch.size() != model.dim) {
    throw std::invalid_argument("PQ distance-table buffer shape mismatch");
  }
  transform(model, input, transformed_scratch);
  const u32 dsub = model.subvector_dim();
  for (u32 subquantizer = 0; subquantizer < model.subquantizers; ++subquantizer) {
    const f32* value = transformed_scratch.data() + static_cast<size_t>(subquantizer) * dsub;
    const f32* centroids = model.centroids.data() +
      static_cast<size_t>(subquantizer) * kCentroidsPerSubquantizer * dsub;
    for (u32 centroid = 0; centroid < kCentroidsPerSubquantizer; ++centroid) {
      const f32* candidate = centroids + static_cast<size_t>(centroid) * dsub;
      const f32 distance = squared_l2_saturated(value, candidate, dsub);
      table[static_cast<size_t>(subquantizer) * kCentroidsPerSubquantizer + centroid] =
        distance;
    }
  }
}

f32 asymmetric_distance(const Model& model, std::span<const f32> table,
                        std::span<const u8> code) {
  if (table.size() != static_cast<size_t>(model.subquantizers) *
        kCentroidsPerSubquantizer || code.size() != model.code_bytes()) {
    throw std::invalid_argument("PQ asymmetric-distance buffer shape mismatch");
  }
  f64 distance = 0.0;
  for (u32 subquantizer = 0; subquantizer < model.subquantizers; ++subquantizer) {
    distance += static_cast<f64>(table[static_cast<size_t>(subquantizer) *
      kCentroidsPerSubquantizer + code[subquantizer]]);
  }
  return saturate_squared_l2(distance);
}

}  // namespace gpu_search::pq
