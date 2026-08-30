#include "tools/vamana_offline/dataset_io.hh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

#include <library/utils.hh>

#include "tools/vamana_offline/progress.hh"

namespace tools::vamana_offline {

namespace {

u32 read_u32(std::ifstream& input) {
  u32 value{};
  if (!input.read(reinterpret_cast<char*>(&value), sizeof(value))) {
    lib_failure("failed to read u32 from dataset");
  }
  return value;
}

float l2_float_query_to_raw(const float* query, const byte_t* rhs, VectorDType dtype, u32 dim) {
  return typed_l2_distance_float_query(
    span<const f32>{query, dim}, rhs, dtype, dim);
}

}  // namespace

filepath_t resolve_dataset_file(const filepath_t& input_path) {
  if (std::filesystem::is_regular_file(input_path)) return input_path;
  if (!std::filesystem::is_directory(input_path)) return input_path;
  static const vec<str> candidates = {"base.fbin", "base.u8bin", "base.i8bin", "base.bin"};
  for (const auto& c : candidates) {
    const filepath_t path = input_path / c;
    if (std::filesystem::exists(path)) return path;
  }
  lib_failure("unable to resolve dataset file under " + input_path.string());
  return {};
}

Dataset read_dataset(const VamanaBuildConfig& config) {
  Dataset dataset;
  dataset.source_file = resolve_dataset_file(config.data_path);

  std::ifstream input(dataset.source_file, std::ios::binary);
  lib_assert(input.good(), "dataset file does not exist: " + dataset.source_file.string());

  const str ext = dataset.source_file.extension().string();
  const auto inferred_dtype = infer_vector_dtype_from_path(dataset.source_file);
  lib_assert(inferred_dtype.has_value(), "unsupported dataset extension: " + ext);
  dataset.dtype = resolve_vector_dtype_config(config.vector_data_type, dataset.source_file);
  if (config.vector_data_type != "auto" && inferred_dtype.has_value() && dataset.dtype != *inferred_dtype) {
    lib_failure("--vector-data-type=" + config.vector_data_type +
                " does not match dataset suffix " + ext +
                " (inferred " + vector_dtype_name(*inferred_dtype) + ")");
  }

  dataset.total_vectors = read_u32(input);
  dataset.dim = read_u32(input);
  dataset.vector_bytes = vector_dtype_bytes(dataset.dtype, dataset.dim);
  lib_assert(dataset.dim > 0, "dataset dimension must be > 0");
  lib_assert(dataset.vector_bytes > 0, "dataset vector byte size must be > 0");
  lib_assert(static_cast<size_t>(dataset.total_vectors) <=
                 (std::numeric_limits<size_t>::max() - 2 * sizeof(u32)) /
                     dataset.vector_bytes,
             "dataset file size calculation overflows");
  const size_t expected_file_bytes = 2 * sizeof(u32) +
      static_cast<size_t>(dataset.total_vectors) * dataset.vector_bytes;
  std::error_code file_size_error;
  const auto actual_file_bytes =
      std::filesystem::file_size(dataset.source_file, file_size_error);
  lib_assert(!file_size_error,
             "cannot stat dataset file: " + dataset.source_file.string());
  lib_assert(actual_file_bytes == expected_file_bytes,
             "dataset file size does not match its header: expected " +
                 std::to_string(expected_file_bytes) + ", got " +
                 std::to_string(actual_file_bytes));
  dataset.vector_count = std::min(dataset.total_vectors, config.max_vectors);
  lib_assert(dataset.vector_count <=
                 std::numeric_limits<size_t>::max() / dataset.vector_bytes,
             "dataset memory allocation size overflows");
  lib_assert(dataset.vector_count > 0, "dataset is empty");
  lib_assert(dataset.vector_count <= static_cast<size_t>(std::numeric_limits<u32>::max()),
             "offline builder currently supports at most 2^32-1 vectors");

  std::cerr << "reading dataset " << dataset.source_file
            << " (dim=" << dataset.dim << ", vectors=" << dataset.vector_count
            << "/" << dataset.total_vectors << ", dtype=" << vector_dtype_name(dataset.dtype)
            << ", vector_bytes=" << dataset.vector_bytes << ")\n";

  dataset.raw_vectors.resize(dataset.vector_count * dataset.vector_bytes);
  ProgressReporter progress{"Reading raw dataset", dataset.vector_count};
  const size_t rows_per_chunk = std::max<size_t>(1, (64 * 1024 * 1024) / std::max<size_t>(1, dataset.vector_bytes));
  for (size_t row = 0; row < dataset.vector_count; row += rows_per_chunk) {
    const size_t chunk_rows = std::min(rows_per_chunk, dataset.vector_count - row);
    byte_t* raw_dst = dataset.raw_vectors.data() + row * dataset.vector_bytes;
    const size_t chunk_bytes = chunk_rows * dataset.vector_bytes;
    if (!input.read(reinterpret_cast<char*>(raw_dst), static_cast<std::streamsize>(chunk_bytes))) {
      lib_failure("failed to read dataset payload");
    }
    progress.increment(chunk_rows);
  }
  if (dataset.dtype == VectorDType::float32) {
    const auto* components = reinterpret_cast<const f32*>(
      dataset.raw_vectors.data());
    const size_t component_count = dataset.vector_count * dataset.dim;
    for (size_t index = 0; index < component_count; ++index) {
      if (!floating_value_is_finite(components[index])) {
        lib_failure("float32 dataset contains a non-finite component");
      }
    }
  }
  progress.finish();

  std::cerr << "offline dataset memory: raw_vectors=" << dataset.raw_vectors.size()
            << " bytes, float_expansion=0 bytes\n";
  return dataset;
}

float dataset_l2_distance(const Dataset& dataset, size_t lhs, size_t rhs) {
  const byte_t* a = dataset.raw_vector(lhs);
  const byte_t* b = dataset.raw_vector(rhs);
  return typed_l2_distance(
    a, dataset.dtype, b, dataset.dtype, dataset.dim);
}

float dataset_distance(const Dataset& dataset, size_t lhs, size_t rhs) {
  return dataset_l2_distance(dataset, lhs, rhs);
}

float dataset_distance_float_query(const Dataset& dataset, const float* query, size_t rhs) {
  return l2_float_query_to_raw(query, dataset.raw_vector(rhs), dataset.dtype, dataset.dim);
}

void dataset_decode_vector(const Dataset& dataset, size_t row, float* dst) {
  decode_storage_vector_to_float(dataset.raw_vector(row), dataset.dtype, dataset.dim, dst);
}

}  // namespace tools::vamana_offline
