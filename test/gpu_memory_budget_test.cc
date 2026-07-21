#include <cassert>
#include <cstdint>
#include <limits>

#include "gpu_search/memory_budget.hh"

namespace {

constexpr std::uint64_t gib(std::uint64_t value) {
  return value << 30;
}

gpu_search::memory_budget::Request sift_request(std::uint64_t nodes) {
  return {
    .nodes = nodes,
    .usable_bytes = gib(36),
    .dim = 128,
    .pq_subquantizers = 16,
    .code_bytes = 16,
    .query_slots = 256,
    .beam_width = 64,
    .graph_degree = 96,
    .exact_width = 256,
    .exact_record_bytes = 144,
    .shard_count = 5,
  };
}

}  // namespace

int main() {
  const auto sift100m =
    gpu_search::memory_budget::estimate(sift_request(100'000'000));
  assert(sift100m.fits);
  assert(sift100m.code_bytes == 1'600'000'000ULL);
  assert(sift100m.exact_bytes == 256ULL * 256 * 144);
  assert(sift100m.fixed_bytes == sift100m.explicit_bytes);
  assert(sift100m.explicit_bytes <= gib(36));
  assert(sift100m.visited_capacity >= 64 * 96 * 8);

  const auto sift1b =
    gpu_search::memory_budget::estimate(sift_request(1'000'000'000));
  assert(sift1b.fits);
  assert(sift1b.code_bytes == 16'000'000'000ULL);
  assert(sift1b.explicit_bytes <= gib(36));

  auto sift1b_pq32_request = sift_request(1'000'000'000);
  sift1b_pq32_request.pq_subquantizers = 32;
  sift1b_pq32_request.code_bytes = 32;
  sift1b_pq32_request.beam_width = 128;
  const auto sift1b_pq32 =
    gpu_search::memory_budget::estimate(sift1b_pq32_request);
  assert(sift1b_pq32.fits);
  assert(sift1b_pq32.code_bytes == 32'000'000'000ULL);
  assert(sift1b_pq32.explicit_bytes <= gib(36));
  assert(sift1b_pq32.query_workspace_bytes > sift1b.query_workspace_bytes);

  auto undersized = sift1b_pq32_request;
  undersized.usable_bytes = gib(16);
  assert(!gpu_search::memory_budget::estimate(undersized).fits);

  auto invalid = sift_request(100'000'000);
  invalid.code_bytes = 32;
  assert(!gpu_search::memory_budget::estimate(invalid).fits);

  // The public layout dimension is u32. Budget arithmetic must reject an
  // impossible OPQ matrix instead of wrapping it into a small allocation.
  auto overflowing = sift_request(100'000'000);
  overflowing.usable_bytes = std::numeric_limits<std::uint64_t>::max();
  overflowing.dim = std::numeric_limits<std::uint32_t>::max();
  overflowing.pq_subquantizers = 1;
  overflowing.code_bytes = 1;
  const auto overflow_result =
    gpu_search::memory_budget::estimate(overflowing);
  assert(!overflow_result.fits);
  assert(overflow_result.explicit_bytes == 0);

  gpu_search::QueryDescriptor descriptor{};
  descriptor.dim = 70'000;
  assert(descriptor.dim == 70'000);

  assert(gpu_search::persistent_score_chunk_capacity(68, 128) == 16);
  // R=128 has eight bounded provisional slots. It remains supported by
  // processing fourteen expansions per scoring chunk instead of overflowing
  // the 2048-item merge workspace at the old fixed width of sixteen.
  assert(gpu_search::persistent_score_chunk_capacity(136, 128) == 14);
  assert(gpu_search::persistent_score_chunk_capacity(2048, 128) == 0);
  return 0;
}
