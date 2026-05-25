#pragma once

#include <Eigen/Dense>

#include "tools/vamana_offline/dataset_io.hh"

namespace tools::vamana_offline {

struct RaBitQState {
  Eigen::MatrixXf rotation_matrix;  // dim x dim, column-major
  vec<float> rotated_centroid;       // dim
  double t_const{0.0};

double best_rescale_factor(const float* abs_unit_residual, size_t dim, size_t bits_per_dim);
double get_const_scaling_factors(size_t dim, size_t bits_per_dim, uint64_t seed);
RaBitQState init_rabitq(const Dataset& dataset, u32 bits_per_dim, int seed);
void rabitq_quantize_vector(const float* vector, const RaBitQState& state, byte_t* output);

}  // namespace tools::vamana_offline
