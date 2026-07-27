#include <cassert>
#include <cmath>
#include <filesystem>
#include <vector>

#include "gpu_search/pq_index.hh"

int main() {
  gpu_search::pq::Model model;
  model.dim = 4;
  model.subquantizers = 2;
  model.centroids.resize(2 * 256 * 2, 1000.0f);
  for (u32 subquantizer = 0; subquantizer < 2; ++subquantizer) {
    for (u32 centroid = 0; centroid < 256; ++centroid) {
      const size_t base = (static_cast<size_t>(subquantizer) * 256 + centroid) * 2;
      model.centroids[base] = static_cast<f32>(centroid);
      model.centroids[base + 1] = static_cast<f32>(centroid * 2);
    }
  }
  std::string error;
  assert(gpu_search::pq::validate(model, &error));

  const std::vector<f32> query{3.0f, 6.0f, 7.0f, 14.0f};
  std::vector<f32> scratch(4);
  std::vector<u8> code(2);
  gpu_search::pq::encode(model, query, code, scratch);
  assert(code[0] == 3 && code[1] == 7);

  std::vector<f32> table(2 * 256);
  gpu_search::pq::build_distance_table(model, query, table, scratch);
  assert(std::abs(gpu_search::pq::asymmetric_distance(model, table, code)) < 1e-6f);

  const auto path = std::filesystem::temp_directory_path() / "dvstor-pq16-model.bin";
  assert(gpu_search::pq::write_model(path, model, &error));
  gpu_search::pq::Model loaded;
  assert(gpu_search::pq::read_model(path, loaded, &error));
  assert(loaded.dim == model.dim);
  assert(loaded.subquantizers == model.subquantizers);
  assert(loaded.centroids == model.centroids);
  assert(loaded.checksum() == model.checksum());
  std::filesystem::remove(path);
  return 0;
}
