#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

#include "common/types.hh"

namespace vamana::routing {

inline constexpr size_t kLiveSeedExplorationSamples = 4;
inline constexpr size_t kLiveSeedWordProbeBudget = 8;

struct CentroidSeedRank {
  long double squared_l2{};
  u64 pointer_raw{};
};

inline bool centroid_seed_rank_less(const CentroidSeedRank& lhs,
                                    const CentroidSeedRank& rhs) {
  if (lhs.squared_l2 != rhs.squared_l2) {
    return lhs.squared_l2 < rhs.squared_l2;
  }
  return lhs.pointer_raw < rhs.pointer_raw;
}

// long double keeps the comparison finite for every finite float32/int8/u8
// component supported by the index, including dimensions whose float64
// squared-L2 sum would overflow. The routing centroid itself remains
// compensated FP64 state; extended precision is used only for deterministic
// ranking.
inline long double centroid_seed_squared_l2(span<const f32> vector,
                                            span<const f64> centroid) {
  if (vector.size() != centroid.size() || vector.empty()) {
    return std::numeric_limits<long double>::infinity();
  }
  long double result = 0;
  for (size_t dimension = 0; dimension < vector.size(); ++dimension) {
    const long double difference =
      static_cast<long double>(vector[dimension]) -
      static_cast<long double>(centroid[dimension]);
    result += difference * difference;
  }
  return result;
}

// Return at most candidate_probe_budget set-bit ordinals while inspecting at
// most word_probe_budget bitmap chunks. The cursor always advances past the
// inspected range, including empty/sparse words, so repeated membership
// batches eventually explore the whole live population without any one batch
// doing work proportional to the dataset size.
inline void bounded_rotating_live_samples_into(
    span<const u64> bitmap,
    u64 valid_bits,
    u64& cursor,
    size_t candidate_probe_budget,
    vec<u64>& samples,
    size_t word_probe_budget = kLiveSeedWordProbeBudget) {
  samples.clear();
  if (bitmap.empty() || valid_bits == 0 || candidate_probe_budget == 0 ||
      word_probe_budget == 0) {
    return;
  }
  const u64 available_bits = std::min<u64>(
    valid_bits, static_cast<u64>(bitmap.size()) * 64);
  if (available_bits == 0) return;
  cursor %= available_bits;
  samples.reserve(candidate_probe_budget);

  u64 position = cursor;
  const size_t bitmap_words = static_cast<size_t>(
    (available_bits + 63) / 64);
  // Starting in the middle of a word and later wrapping to its prefix may
  // inspect that physical word twice; the +1 accounts for that split chunk.
  const size_t chunk_budget = std::min(
    word_probe_budget,
    bitmap_words + static_cast<size_t>((cursor % 64) != 0));
  for (size_t chunk = 0;
       chunk < chunk_budget && samples.size() < candidate_probe_budget;
       ++chunk) {
    const size_t word_index = static_cast<size_t>(position / 64);
    const u32 first_bit = static_cast<u32>(position % 64);
    u64 word = bitmap[word_index];
    if (first_bit != 0) word &= ~((u64{1} << first_bit) - 1);
    if (word_index + 1 == bitmap_words && available_bits % 64 != 0) {
      word &= (u64{1} << (available_bits % 64)) - 1;
    }
    while (word != 0 && samples.size() < candidate_probe_budget) {
      const u32 bit = static_cast<u32>(std::countr_zero(word));
      const u64 ordinal = static_cast<u64>(word_index) * 64 + bit;
      samples.push_back(ordinal);
      word &= word - 1;
      position = ordinal + 1 == available_bits ? 0 : ordinal + 1;
      cursor = position;
    }
    if (word == 0) {
      const u64 next_word = static_cast<u64>(word_index + 1) * 64;
      position = next_word >= available_bits ? 0 : next_word;
      cursor = position;
    }
  }
}

inline vec<u64> bounded_rotating_live_samples(
    span<const u64> bitmap,
    u64 valid_bits,
    u64& cursor,
    size_t candidate_probe_budget,
    size_t word_probe_budget = kLiveSeedWordProbeBudget) {
  vec<u64> samples;
  bounded_rotating_live_samples_into(
    bitmap, valid_bits, cursor, candidate_probe_budget, samples,
    word_probe_budget);
  return samples;
}

}  // namespace vamana::routing
