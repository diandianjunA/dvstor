#include <cassert>
#include <cstdint>

#include "gpu_search/index_format.hh"
#include "gpu_search/memory_budget.hh"

namespace {

constexpr u64 gib(u64 value) {
  return value << 30;
}

gpu_search::memory_budget::Request sift_request(u64 nodes) {
  return {
    .nodes = nodes,
    .max_delta_vectors = nodes,
    .usable_bytes = gib(36),
    .requested_cache_bytes = gib(3),
    .requested_exact_cache_bytes = gib(4),
    .delta_budget_bytes = gib(2),
    .dim = 128,
    .pq_subquantizers = 16,
    .code_bytes = 16,
    .vector_bytes = 128,
    .query_slots = 256,
    .beam_width = 64,
    .graph_degree = 96,
    .exact_width = 256,
    .exact_record_bytes = 136,
    .anchor_count = 4096 * 5,
    .shard_count = 5,
    .entry_point_count = 256,
    .cache_ways = 4,
    .exact_cache_ways = 4,
  };
}

}  // namespace

int main() {
  const auto sift100m = gpu_search::memory_budget::estimate(sift_request(100'000'000));
  assert(sift100m.fits);
  assert(sift100m.code_bytes == 1'600'000'000ULL);
  assert(sift100m.delta_bytes <= gib(2));
  assert(sift100m.delta_code_bytes ==
         static_cast<u64>(sift100m.delta_capacity) * 16);
  assert(sift100m.delta_code_bytes < sift100m.delta_bytes);
  assert(sift100m.delta_capacity > 7'000'000);
  assert(sift100m.cache_total_bytes <= gib(3));
  assert(sift100m.exact_cache_total_bytes <= gib(4));
  assert(sift100m.explicit_bytes <= gib(36));

  const auto sift1b = gpu_search::memory_budget::estimate(sift_request(1'000'000'000));
  assert(sift1b.fits);
  assert(sift1b.code_bytes == 16'000'000'000ULL);
  assert(sift1b.delta_bytes <= gib(2));
  assert(sift1b.delta_code_bytes ==
         static_cast<u64>(sift1b.delta_capacity) * 16);
  assert(sift1b.cache_total_bytes <= gib(3));
  assert(sift1b.exact_cache_total_bytes <= gib(4));
  assert(sift1b.explicit_bytes <= gib(36));
  assert(sift1b.cache_slots > 1'000'000);
  assert(sift1b.exact_cache_slots > 1'000'000);

  auto sift1b_pq32_request = sift_request(1'000'000'000);
  sift1b_pq32_request.pq_subquantizers = 32;
  sift1b_pq32_request.code_bytes = 32;
  sift1b_pq32_request.beam_width = 128;
  const auto sift1b_pq32 = gpu_search::memory_budget::estimate(sift1b_pq32_request);
  assert(sift1b_pq32.fits);
  assert(sift1b_pq32.code_bytes == 32'000'000'000ULL);
  assert(sift1b_pq32.delta_bytes <= gib(2));
  assert(sift1b_pq32.cache_total_bytes <= gib(3));
  assert(sift1b_pq32.exact_cache_total_bytes < gib(4));
  assert(sift1b_pq32.cache_sets >= sift1b_pq32_request.query_slots);
  assert(sift1b_pq32.exact_cache_sets >= sift1b_pq32_request.query_slots);
  assert(sift1b_pq32.explicit_bytes <= gib(36));

  const u64 pq_model_bytes =
    (128ull * 128 + 128ull * 256) * sizeof(f32);
  const u64 compute_local_files = pq_model_bytes;
  assert(compute_local_files < (1ull << 20));
  assert(compute_local_files < gib(50));

  auto undersized = sift_request(1'000'000'000);
  undersized.usable_bytes = gib(16);
  assert(!gpu_search::memory_budget::estimate(undersized).fits);
  return 0;
}
