#include "tools/vamana_offline/dataset_io.hh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

#include <library/utils.hh>

#include "tools/vamana_offline/progress.hh"

namespace tools::vamana_offline {

u32 read_u32(std::ifstream& input) {
  u32 value{};
  if (!input.read(reinterpret_cast<char*>(&value), sizeof(value)))
    lib_failure("failed to read u32 from dataset");
  return value;
}

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
  const bool is_float32 = ext == ".fbin" || ext == ".bin";
  const bool is_uint8 = ext == ".u8bin";
  const bool is_int8 = ext == ".i8bin";
  lib_assert(is_float32 || is_uint8 || is_int8, "unsupported dataset extension: " + ext);

  dataset.total_vectors = read_u32(input);
  dataset.dim = read_u32(input);

  const size_t num_vectors = std::min(dataset.total_vectors, config.max_vectors);
  lib_assert(num_vectors > 0, "dataset is empty");

  std::cerr << "reading dataset " << dataset.source_file
            << " (dim=" << dataset.dim << ", vectors=" << num_vectors
            << "/" << dataset.total_vectors << ")\n";

  dataset.vectors.resize(num_vectors * dataset.dim);
  dataset.ids.resize(num_vectors);
  std::iota(dataset.ids.begin(), dataset.ids.end(), 0);

  if (is_float32) {
    ProgressReporter progress{"Reading dataset", num_vectors};
    const size_t rows_per_chunk = std::max<size_t>(1, (8 * 1024 * 1024) / (dataset.dim * sizeof(element_t)));
    for (size_t row = 0; row < num_vectors; row += rows_per_chunk) {
      const size_t chunk_rows = std::min(rows_per_chunk, num_vectors - row);
      const size_t chunk_bytes = chunk_rows * dataset.dim * sizeof(element_t);
      if (!input.read(reinterpret_cast<char*>(dataset.vectors.data() + row * dataset.dim), chunk_bytes))
        lib_failure("failed to read float32 dataset payload");
      progress.increment(chunk_rows);
    }
    progress.finish();
  } else if (is_uint8) {
    vec<u8> raw(dataset.vectors.size());
    if (!input.read(reinterpret_cast<char*>(raw.data()), raw.size()))
      lib_failure("failed to read uint8 dataset payload");
    ProgressReporter progress{"Converting dataset", num_vectors};
    parallel_for(0, num_vectors, config.threads, [&](size_t row, size_t) {
      const size_t base = row * dataset.dim;
      for (size_t col = 0; col < dataset.dim; ++col)
        dataset.vectors[base + col] = static_cast<element_t>(raw[base + col]);
      progress.increment();
    });
    progress.finish();
  } else {
    vec<i8> raw(dataset.vectors.size());
    if (!input.read(reinterpret_cast<char*>(raw.data()), raw.size()))
      lib_failure("failed to read int8 dataset payload");
    ProgressReporter progress{"Converting dataset", num_vectors};
    parallel_for(0, num_vectors, config.threads, [&](size_t row, size_t) {
      const size_t base = row * dataset.dim;
      for (size_t col = 0; col < dataset.dim; ++col)
        dataset.vectors[base + col] = static_cast<element_t>(raw[base + col]);
      progress.increment();
    });
    progress.finish();
  }

  return dataset;
}


}  // namespace tools::vamana_offline
