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
  float sum = 0.0f;
  for (u32 i = 0; i < dim; ++i) {
    const float diff = query[i] - vector_component_as_float(rhs, dtype, i);
    sum += diff * diff;
  }
  return sum;
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
  dataset.vector_count = std::min(dataset.total_vectors, config.max_vectors);
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
  progress.finish();

  std::cerr << "offline dataset memory: raw_vectors=" << dataset.raw_vectors.size()
            << " bytes, float_expansion=0 bytes\n";
  return dataset;
}

float dataset_l2_distance(const Dataset& dataset, size_t lhs, size_t rhs) {
  const byte_t* a = dataset.raw_vector(lhs);
  const byte_t* b = dataset.raw_vector(rhs);
  const u32 dim = dataset.dim;
  switch (dataset.dtype) {
    case VectorDType::uint8: {
      const auto* au = reinterpret_cast<const u8*>(a);
      const auto* bu = reinterpret_cast<const u8*>(b);
      u32 sum = 0;
      for (u32 i = 0; i < dim; ++i) {
        const int diff = static_cast<int>(au[i]) - static_cast<int>(bu[i]);
        sum += static_cast<u32>(diff * diff);
      }
      return static_cast<float>(sum);
    }
    case VectorDType::int8: {
      const auto* ai = reinterpret_cast<const i8*>(a);
      const auto* bi = reinterpret_cast<const i8*>(b);
      u32 sum = 0;
      for (u32 i = 0; i < dim; ++i) {
        const int diff = static_cast<int>(ai[i]) - static_cast<int>(bi[i]);
        sum += static_cast<u32>(diff * diff);
      }
      return static_cast<float>(sum);
    }
    case VectorDType::float32:
      return typed_l2_distance(a, dataset.dtype, b, dataset.dtype, dim);
  }
  return 0.0f;
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
