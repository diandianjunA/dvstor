#include "dvstor_index.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

template <class T>
T read_scalar(std::ifstream& input, const std::string& label) {
  T value{};
  if (!input.read(reinterpret_cast<char*>(&value), sizeof(T))) {
    throw std::runtime_error("failed to read " + label);
  }
  return value;
}

struct Matrix {
  uint32_t rows{};
  uint32_t dim{};
  std::vector<float> values;

  const float* row(size_t i) const {
    return values.data() + i * dim;
  }
};

Matrix read_fbin(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open fbin file: " + path);
  }

  Matrix matrix;
  matrix.rows = read_scalar<uint32_t>(input, "fbin row count");
  matrix.dim = read_scalar<uint32_t>(input, "fbin dim");
  matrix.values.resize(static_cast<size_t>(matrix.rows) * matrix.dim);
  if (!input.read(reinterpret_cast<char*>(matrix.values.data()),
                  static_cast<std::streamsize>(matrix.values.size() * sizeof(float)))) {
    throw std::runtime_error("failed to read fbin payload: " + path);
  }
  return matrix;
}

struct GroundTruth {
  uint32_t rows{};
  uint32_t top_k{};
  std::vector<uint32_t> ids;

  const uint32_t* row(size_t i) const {
    return ids.data() + i * top_k;
  }
};

GroundTruth read_groundtruth(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open groundtruth file: " + path);
  }

  GroundTruth gt;
  gt.rows = read_scalar<uint32_t>(input, "groundtruth row count");
  gt.top_k = read_scalar<uint32_t>(input, "groundtruth top_k");
  gt.ids.resize(static_cast<size_t>(gt.rows) * gt.top_k);
  if (!input.read(reinterpret_cast<char*>(gt.ids.data()),
                  static_cast<std::streamsize>(gt.ids.size() * sizeof(uint32_t)))) {
    throw std::runtime_error("failed to read groundtruth payload: " + path);
  }
  return gt;
}

std::vector<float> copy_row(const Matrix& matrix, size_t row) {
  const float* begin = matrix.row(row);
  return {begin, begin + matrix.dim};
}

double recall_at(const std::vector<uint32_t>& results, const uint32_t* gt, uint32_t k) {
  size_t hits = 0;
  for (uint32_t id : results) {
    for (uint32_t i = 0; i < k; ++i) {
      if (id == gt[i]) {
        ++hits;
        break;
      }
    }
  }
  return static_cast<double>(hits) / static_cast<double>(k);
}

double run_recall_check(DvstorIndex& index,
                        const Matrix& queries,
                        const GroundTruth& gt,
                        uint32_t k,
                        const std::string& label) {
  if (queries.rows != gt.rows) {
    throw std::runtime_error("query/groundtruth row count mismatch");
  }
  if (k == 0 || k > gt.top_k) {
    throw std::runtime_error("invalid recall k");
  }

  double total_recall = 0.0;
  size_t self_hits = 0;
  std::vector<uint32_t> ids;
  std::vector<float> distances;
  for (uint32_t qi = 0; qi < queries.rows; ++qi) {
    const std::vector<float> query = copy_row(queries, qi);
    ids.clear();
    distances.clear();
    index.search(query, k, ids, distances);
    total_recall += recall_at(ids, gt.row(qi), k);
    if (std::find(ids.begin(), ids.end(), gt.row(qi)[0]) != ids.end()) {
      ++self_hits;
    }
  }

  const double recall = total_recall / static_cast<double>(queries.rows);
  const double self_hit_rate = static_cast<double>(self_hits) / static_cast<double>(queries.rows);
  std::cout << "[offline-layout] " << label
            << " recall@" << k << "=" << recall
            << " self_hit_rate=" << self_hit_rate
            << " queries=" << queries.rows << std::endl;
  return recall;
}

std::vector<float> make_update_vectors(size_t count, size_t dim) {
  std::vector<float> vectors(count * dim, 0.0f);
  for (size_t i = 0; i < count; ++i) {
    for (size_t d = 0; d < dim; ++d) {
      vectors[i * dim + d] = 50.0f + static_cast<float>((i * 17 + d * 13) % 97) / 97.0f;
    }
    vectors[i * dim + (i % dim)] += 8.0f;
  }
  return vectors;
}

void run_update_check(DvstorIndex& index, size_t dim) {
  constexpr size_t update_count = 16;
  constexpr uint32_t update_id_base = 1000000;
  const std::vector<float> vectors = make_update_vectors(update_count, dim);
  std::vector<uint32_t> ids(update_count);
  for (size_t i = 0; i < update_count; ++i) {
    ids[i] = update_id_base + static_cast<uint32_t>(i);
  }

  const size_t inserted = index.insert_count(vectors, ids);
  std::cout << "[offline-layout] update inserted=" << inserted
            << " expected=" << update_count << std::endl;
  if (inserted != update_count) {
    throw std::runtime_error("update insert count mismatch");
  }

  size_t misses = 0;
  std::vector<uint32_t> result_ids;
  std::vector<float> result_distances;
  for (size_t i = 0; i < update_count; ++i) {
    const std::vector<float> query(vectors.begin() + static_cast<std::ptrdiff_t>(i * dim),
                                   vectors.begin() + static_cast<std::ptrdiff_t>((i + 1) * dim));
    result_ids.clear();
    result_distances.clear();
    index.search(query, 10, result_ids, result_distances);
    if (std::find(result_ids.begin(), result_ids.end(), ids[i]) == result_ids.end()) {
      ++misses;
      std::cerr << "[offline-layout] missing inserted id " << ids[i] << " results=";
      for (uint32_t id : result_ids) {
        std::cerr << id << ' ';
      }
      std::cerr << std::endl;
    }
  }

  std::cout << "[offline-layout] update self-query misses=" << misses << std::endl;
  if (misses != 0) {
    throw std::runtime_error("inserted-vector self-query failed");
  }
}

double parse_threshold(int argc, char** argv) {
  if (argc < 6) {
    return 0.60;
  }
  return std::stod(argv[5]);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " <service_config> <base.fbin> <query.fbin> <groundtruth.bin> [min_recall]"
              << std::endl;
    return EXIT_FAILURE;
  }

  try {
    const std::string config_path = argv[1];
    const std::string base_path = argv[2];
    const std::string query_path = argv[3];
    const std::string gt_path = argv[4];
    const double min_recall = parse_threshold(argc, argv);

    const Matrix base = read_fbin(base_path);
    const Matrix queries = read_fbin(query_path);
    const GroundTruth gt = read_groundtruth(gt_path);
    if (base.dim != queries.dim) {
      throw std::runtime_error("base/query dimension mismatch");
    }

    DvstorIndex index(config_path);
    if (index.dimension() != base.dim) {
      throw std::runtime_error("service dimension mismatch");
    }

    const uint32_t recall_k = std::min<uint32_t>(10, gt.top_k);
    const double before = run_recall_check(index, queries, gt, recall_k, "before-update");
    if (before < min_recall) {
      throw std::runtime_error("offline index recall below threshold before update");
    }

    run_update_check(index, base.dim);

    const double after = run_recall_check(index, queries, gt, recall_k, "after-update");
    if (after < min_recall) {
      throw std::runtime_error("offline index recall below threshold after update");
    }

    std::cout << "[offline-layout] passed" << std::endl;
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "DvstorOfflineLayoutSmokeTest failed: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
