#include "tools/vamana_offline/rabitq.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <queue>
#include <random>

namespace tools::vamana_offline {

constexpr std::array<float, 9> kTightStart = {
  0.0f,
  0.15f,
  0.20f,
  0.52f,
  0.59f,
  0.71f,
  0.75f,
  0.77f,
  0.81f
};

double best_rescale_factor(const float* abs_unit_residual, size_t dim, size_t bits_per_dim) {
  constexpr double kEps = 1e-5;
  constexpr int kNEnum = 10;

  const double max_o = *std::max_element(abs_unit_residual, abs_unit_residual + dim);
  if (max_o <= 0.0) {
    return 0.0;
  }

  const double t_end = static_cast<double>(((1u << bits_per_dim) - 1u) + kNEnum) / max_o;
  const double t_start = t_end * kTightStart.at(bits_per_dim);

  vec<int> cur_o_bar(dim);
  double sqr_denominator = static_cast<double>(dim) * 0.25;
  double numerator = 0.0;

  for (size_t i = 0; i < dim; ++i) {
    const int cur = static_cast<int>((t_start * abs_unit_residual[i]) + kEps);
    cur_o_bar[i] = cur;
    sqr_denominator += static_cast<double>(cur) * cur + cur;
    numerator += (static_cast<double>(cur) + 0.5) * abs_unit_residual[i];
  }

  std::priority_queue<std::pair<double, size_t>,
                      vec<std::pair<double, size_t>>,
                      std::greater<>> next_t;
  for (size_t i = 0; i < dim; ++i) {
    if (abs_unit_residual[i] > 0.0f) {
      next_t.emplace(static_cast<double>(cur_o_bar[i] + 1) / abs_unit_residual[i], i);
    }
  }

  double max_ip = 0.0;
  double best_t = 0.0;
  while (!next_t.empty()) {
    const auto [cur_t, update_id] = next_t.top();
    next_t.pop();

    cur_o_bar[update_id]++;
    const int update_o_bar = cur_o_bar[update_id];
    sqr_denominator += 2.0 * update_o_bar;
    numerator += abs_unit_residual[update_id];

    const double cur_ip = numerator / std::sqrt(sqr_denominator);
    if (cur_ip > max_ip) {
      max_ip = cur_ip;
      best_t = cur_t;
    }

    if (update_o_bar < static_cast<int>((1u << bits_per_dim) - 1u)) {
      const double t_next = static_cast<double>(update_o_bar + 1) / abs_unit_residual[update_id];
      if (t_next < t_end) {
        next_t.emplace(t_next, update_id);
      }
    }
  }

  return best_t;
}

double get_const_scaling_factors(size_t dim, size_t bits_per_dim, uint64_t seed) {
  constexpr size_t n_samples = 1000;
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  vec<float> sample(dim);
  vec<float> abs_unit(dim);
  double total = 0.0;

  for (size_t sample_id = 0; sample_id < n_samples; ++sample_id) {
    double norm_sqr = 0.0;
    for (size_t d = 0; d < dim; ++d) {
      sample[d] = normal(rng);
      norm_sqr += static_cast<double>(sample[d]) * sample[d];
    }
    const double norm = std::sqrt(norm_sqr);
    const double inv_norm = norm > 0.0 ? (1.0 / norm) : 0.0;
    for (size_t d = 0; d < dim; ++d) {
      abs_unit[d] = std::fabs(static_cast<float>(sample[d] * inv_norm));
    }
    total += best_rescale_factor(abs_unit.data(), dim, bits_per_dim);
  }

  return total / static_cast<double>(n_samples);
}

/**
 * Initialize RaBitQ: generate random orthogonal matrix via QR decomposition.
 */
RaBitQState init_rabitq(const Dataset& dataset, u32 bits_per_dim, int seed) {
  const u32 dim = dataset.dim;
  const size_t n = dataset.ids.size();

  RaBitQState state;
  state.dim = dim;
  state.bits_per_dim = bits_per_dim;
  state.packed_bytes = (bits_per_dim * dim + 7) / 8;
  state.total_rabitq_bytes = state.packed_bytes + 2 * sizeof(float);

  // Generate random matrix and compute QR decomposition for orthogonal P
  std::cerr << "generating rotation matrix (dim=" << dim << ")...\n";
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  Eigen::MatrixXf random_mat(dim, dim);
  for (u32 i = 0; i < dim; ++i)
    for (u32 j = 0; j < dim; ++j)
      random_mat(i, j) = normal(rng);

  Eigen::HouseholderQR<Eigen::MatrixXf> qr(random_mat);
  state.rotation_matrix = qr.householderQ() * Eigen::MatrixXf::Identity(dim, dim);
  Eigen::MatrixXf r = state.rotation_matrix.transpose() * random_mat;
  for (u32 i = 0; i < dim; ++i) {
    if (r(i, i) < 0.0f) {
      state.rotation_matrix.col(i) *= -1.0f;
    }
  }

  // Compute centroid
  std::cerr << "computing centroid...\n";
  Eigen::VectorXf centroid = Eigen::VectorXf::Zero(dim);
  for (size_t i = 0; i < n; ++i) {
    Eigen::Map<const Eigen::VectorXf> v(dataset.vector(i), dim);
    centroid += v;
  }
  centroid /= static_cast<float>(n);

  // Rotate centroid with the same transform later used by quantization.
  Eigen::VectorXf rot_centroid = state.rotation_matrix.transpose() * centroid;
  state.rotated_centroid.assign(rot_centroid.data(), rot_centroid.data() + dim);

  // Match Jasper's sampled constant scaling factor instead of the earlier simplified approximation.
  state.t_const = get_const_scaling_factors(dim, bits_per_dim, static_cast<uint64_t>(seed));

  return state;
}

/**
 * Quantize a single vector using RaBitQ.
 * Output: [packed_bits(packed_bytes) | add(4B) | rescale(4B)]
 */
void rabitq_quantize_vector(const float* vector,
                            const RaBitQState& state,
                            byte_t* output) {
  const u32 dim = state.dim;
  const u32 bits = state.bits_per_dim;
  constexpr double kEps = 1e-5;

  // Rotate vector: x' = P^T * x
  Eigen::Map<const Eigen::VectorXf> v(vector, dim);
  Eigen::VectorXf rotated = state.rotation_matrix.transpose() * v;

  // Subtract rotated centroid: delta = x' - c'
  vec<float> delta(dim);
  for (u32 i = 0; i < dim; ++i)
    delta[i] = rotated(i) - state.rotated_centroid[i];

  float l2_sqr = 0.0f;
  for (u32 i = 0; i < dim; ++i) l2_sqr += delta[i] * delta[i];
  const float l2_norm = std::sqrt(l2_sqr);

  vec<u8> quantized_vals(dim, 0);
  float ip_norm = 0.0f;
  const u32 magnitude_cap = (1u << (bits - 1)) - 1u;
  for (u32 i = 0; i < dim; ++i) {
    const float abs_o = l2_norm > 0.0f ? std::fabs(delta[i] / l2_norm) : 0.0f;
    int val = static_cast<int>((state.t_const * abs_o) + kEps);
    if (val >= static_cast<int>(1u << (bits - 1))) {
      val = static_cast<int>((1u << (bits - 1)) - 1u);
    }
    quantized_vals[i] = static_cast<u8>(val);
    ip_norm += (static_cast<float>(val) + 0.5f) * abs_o;
  }
  const float ip_norm_inv = ip_norm == 0.0f ? 1.0f : (1.0f / ip_norm);

  for (u32 i = 0; i < dim; ++i) {
    if (delta[i] >= 0.0f) {
      quantized_vals[i] = static_cast<u8>(quantized_vals[i] + (1u << (bits - 1)));
    } else {
      quantized_vals[i] = static_cast<u8>((~quantized_vals[i]) & magnitude_cap);
    }
  }

  // Pack bits into bytes
  std::memset(output, 0, state.packed_bytes);
  for (u32 i = 0; i < dim; ++i) {
    const u32 bit_idx = i * bits;
    const u32 byte_idx = bit_idx / 8;
    const u32 bit_off = bit_idx % 8;
    output[byte_idx] |= static_cast<byte_t>(quantized_vals[i] << bit_off);
    if (bit_off + bits > 8 && byte_idx + 1 < state.packed_bytes) {
      output[byte_idx + 1] |= static_cast<byte_t>(quantized_vals[i] >> (8 - bit_off));
    }
  }

  const float cb = -(static_cast<float>(1u << (bits - 1)) - 0.5f);
  float ip_resi_xucb = 0.0f;
  float ip_cent_xucb = 0.0f;
  for (u32 i = 0; i < dim; ++i) {
    const float xu_cb = static_cast<float>(quantized_vals[i]) + cb;
    ip_resi_xucb += delta[i] * xu_cb;
    ip_cent_xucb += state.rotated_centroid[i] * xu_cb;
  }
  if (ip_resi_xucb == 0.0f) {
    ip_resi_xucb = std::numeric_limits<float>::infinity();
  }

  const float add_factor = l2_sqr + 2.0f * l2_sqr * ip_cent_xucb / ip_resi_xucb;
  const float rescale_factor = ip_norm_inv * -2.0f * l2_norm;

  byte_t* trailer = output + state.packed_bytes;
  std::memcpy(trailer, &add_factor, sizeof(float));
  std::memcpy(trailer + sizeof(float), &rescale_factor, sizeof(float));
}


}  // namespace tools::vamana_offline
